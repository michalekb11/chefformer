import torch
import os
import argparse
import contextlib
from transformers import AutoTokenizer
from configs.training.settings import pretraining_settings
from src.shared.model.model import Chefformer
from src.training.engine import Trainer
from src.training.data_factory import get_dataloaders
from src.training.utils.training_loop_utils import init_weights, lr_schedule
from torch.optim.lr_scheduler import LambdaLR
from loggers.console_logger import ConsoleLogger
from loggers.csv_logger import CSVLogger
from loggers.tensorboard_logger import TensorBoardLogger
from loggers.composite_logger import CompositeMetricLogger
from pydantic_settings import BaseSettings

def train_model(
        task: str, 
        settings: BaseSettings,
        checkpoint_path: str=None,
        eval_only: bool=False,
        profile: bool=False,
    ):
    # Settings look ups
    training_args = settings.training_args

    log_dir = os.path.join("logs", task)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize Loggers
    logger = ConsoleLogger(__name__, log_file=os.path.join(log_dir, "training.log"))
    csv_path = os.path.join(training_args.checkpoint_dir, task, "metrics.csv")
    csv_logger = CSVLogger(csv_path)
    
    # Initialize the composite logger with console and CSV first.
    # Add TensorBoardLogger once we know the start_step.
    metric_logger = CompositeMetricLogger([logger, csv_logger])
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize Data
    train_loader, val_loader = get_dataloaders(task, tokenizer, settings=training_args)

    # Initialize Model & Optimization
    model = Chefformer(settings.model).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = LambdaLR(
        optimizer, 
        lr_lambda=lambda step: lr_schedule(
            step,
            warmup_iters=training_args.warmup_iters,
            decay_start_iter=training_args.decay_start_iter,
            decay_total_iters=training_args.decay_total_iters
        )
    )
    criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)

    trainer = Trainer(
        model, 
        optimizer, 
        scheduler, 
        criterion, 
        device, 
        settings, 
        metric_logger, 
        logger
    )

    # Resume from checkpoint
    start_epoch, start_step = 0, 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, start_step = trainer.load(train_loader, checkpoint_path)
        logger.info(f"Resuming from {checkpoint_path} at epoch {start_epoch}, step {start_step}")
        csv_logger.prepare_for_resume(start_step)
    else:
        logger.info("Starting training from scratch. No checkpoint provided, or checkpoint path does not exist.")
        start_step = 0

    # Now that start_step is known, initialize TensorBoard with the purge_step
    tb_logger = TensorBoardLogger(log_dir=log_dir, purge_step=start_step)
    metric_logger.loggers.append(tb_logger) # Should modify the metric_logger passed into Trainer

    if eval_only:
        avg_loss, acc = trainer.evaluate(val_loader, training_args.validation_loop_steps)
        logger.info(f"Eval results - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
        return

    # Initialize Profiler context
    profiler_context = contextlib.nullcontext()
    if profile:
        profiler_dir = os.path.join(log_dir, "profiler")
        os.makedirs(profiler_dir, exist_ok=True)
        logger.info(f"Profiling enabled. Trace will be saved to: {profiler_dir}")
        profiler_context = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=30, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            record_shapes=False,
            profile_memory=True,
            with_stack=True,
            acc_events=True
        )

    # Main Training Loop
    with profiler_context as prof:
        for epoch in range(start_epoch, training_args.num_epochs):
            for step, (input_ids, attention_mask) in enumerate(train_loader):
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                
                global_step = start_step + step
                loss, acc = trainer.train_step(input_ids, attention_mask, global_step)

                if profile:
                    if device.type == "mps":
                        torch.mps.synchronize()
                    prof.step()
                    if global_step >= start_step + 50:
                        logger.info("Profiling cycle complete. Exiting...")
                        return

                # Periodic Validation
                if (global_step + 1) % training_args.validate_every == 0:
                    val_loss, val_acc = trainer.evaluate(val_loader, training_args.validation_loop_steps, global_step)
                    logger.info(f"Validation at step {global_step+1}: Loss {val_loss:.4f}, Acc {val_acc:.4f}")
                    torch.mps.empty_cache()

                # Periodic Checkpointing
                if (global_step + 1) % training_args.save_checkpoint_every == 0:
                    trainer.save(train_loader, epoch, global_step + 1)
                    torch.mps.empty_cache()

        # End of epoch checkpoint
        trainer.save(train_loader, epoch + 1, 0)
        start_step = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chefformer Training and Evaluation")
    
    parser.add_argument('--task', type=str, required=True, choices=['pretrain'],
                        help='The training task to execute')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to the checkpoint to load (defaults to settings value)')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='If set, the script will only run evaluation (defaults to False)')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Run the PyTorch profiler for a few steps to analyze bottlenecks')
    args = parser.parse_args()

    if args.task == 'pretrain':
        settings = pretraining_settings
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    if not args.checkpoint_path:
        args.checkpoint_path = getattr(settings, 'checkpoint_path', None)

    train_model(
        task=args.task, 
        settings=settings,
        checkpoint_path=args.checkpoint_path, 
        eval_only=args.eval_only,
        profile=args.profile
    )