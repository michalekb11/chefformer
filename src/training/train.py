import torch
import os
import argparse
from transformers import AutoTokenizer
from configs.training.settings import pretraining_settings
from src.shared.model.model import Chefformer
from src.training.engine import Trainer
from src.training.data_factory import get_dataloaders
from src.training.utils.training_loop_utils import init_weights, lr_schedule
from torch.optim.lr_scheduler import LambdaLR
from loggers.console_logger import ConsoleLogger
from loggers.csv_logger import CSVLogger
from loggers.composite_logger import CompositeMetricLogger
from pydantic_settings import BaseSettings

def train_model(
        task: str, 
        settings: BaseSettings,
        checkpoint_path: str=None,
        eval_only: bool=False,
    ):
    # Settings look ups
    training_args = settings.training_args

    # Initialize Loggers
    logger = ConsoleLogger(__name__)
    csv_logger = CSVLogger()
    metric_logger = CompositeMetricLogger([logger, csv_logger]) # Will print to console and save to CSV
    
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
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

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
    else:
        logger.info("Starting training from scratch. No checkpoint provided, or checkpoint path does not exist.")

    if eval_only:
        avg_loss, acc = trainer.evaluate(val_loader, training_args.validation_loop_steps)
        logger.info(f"Eval results - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
        return

    # Main Training Loop
    for epoch in range(start_epoch, training_args.num_epochs):
        for step, (input_ids, attention_mask) in enumerate(train_loader):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            actual_step = start_step + step
            loss, acc = trainer.train_step(input_ids, attention_mask, actual_step)

            # Periodic Validation
            if (actual_step + 1) % training_args.validate_every == 0:
                val_loss, val_acc = trainer.evaluate(val_loader, training_args.validation_loop_steps)
                logger.info(f"Validation at step {actual_step+1}: Loss {val_loss:.4f}, Acc {val_acc:.4f}")

            # Periodic Checkpointing
            if (actual_step + 1) % training_args.save_checkpoint_every == 0:
                trainer.save(train_loader, epoch, actual_step + 1)

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
        eval_only=args.eval_only
    )