import torch
import os
import argparse
import contextlib
from transformers import AutoTokenizer
from configs.training.settings import pretraining_settings, finetuning_settings
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
    if hasattr(settings, 'prompt'):
        base_prompt = settings.prompt.base_prompt
    else:
        base_prompt = None
    train_loader, val_loader = get_dataloaders(
        task, 
        tokenizer, 
        settings=training_args, 
        base_prompt=base_prompt
    )

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
        # Detect if we are loading weights from a different task (e.g., pretrain -> finetune)
        # If the current task name isn't in the checkpoint path, we assume a weights-only transfer.
        weights_only = task not in checkpoint_path
        
        start_epoch, start_step = trainer.load(train_loader, checkpoint_path, weights_only=weights_only)
        
        if weights_only:
            logger.info(f"Transferred weights from {checkpoint_path}. Starting {task} from Epoch 0, Step 0.")
        else:
            logger.info(f"Resuming {task} from {checkpoint_path} at epoch {start_epoch}, step {start_step}")
            csv_logger.prepare_for_resume(start_step)
    else:
        logger.info(f"Starting {task} from scratch. No checkpoint provided, or path does not exist.")

    # Now that start_step is known, initialize TensorBoard with the purge_step
    tb_logger = TensorBoardLogger(log_dir=log_dir, purge_step=start_step + 1)
    metric_logger.loggers.append(tb_logger) # Should modify the metric_logger passed into Trainer

    if eval_only:
        trainer.evaluate(val_loader, start_step, training_args.validation_loop_steps)
        return

    # Initialize Profiler context
    global_step = start_step
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
            for step, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                labels = labels.to(device) if labels is not None else None
                
                loss, acc = trainer.train_step(input_ids, attention_mask, global_step, labels=labels)

                if profile:
                    if device.type == "mps":
                        torch.mps.synchronize()
                    prof.step()
                    if global_step >= start_step + 50:
                        logger.info("Profiling cycle complete. Exiting...")
                        return

                global_step += 1

                # Periodic Validation
                if global_step % training_args.validate_every == 0:
                    trainer.evaluate(
                        val_loader=val_loader, 
                        train_step_count=global_step, 
                        max_steps=training_args.validation_loop_steps
                    )
                    torch.mps.empty_cache()

                # Periodic Checkpointing
                if global_step % training_args.save_checkpoint_every == 0:
                    trainer.save(train_loader, epoch, global_step)
                    torch.mps.empty_cache()

            # End of epoch checkpoint (only save if we didn't JUST save on the last step)
            if global_step % training_args.save_checkpoint_every != 0:
                trainer.save(train_loader, epoch + 1, global_step)
                torch.mps.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chefformer Training and Evaluation")
    
    parser.add_argument('--task', type=str, required=True, choices=['pretrain', 'finetune'],
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
    elif args.task == 'finetune':
        settings = finetuning_settings
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