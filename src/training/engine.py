import os
import torch
from torch.nn.utils import clip_grad_norm_
from loggers.console_logger import ConsoleLogger
from loggers.composite_logger import CompositeMetricLogger

logger = ConsoleLogger(__name__)

class Trainer:
    def __init__(
            self, 
            model, 
            optimizer, 
            scheduler, 
            criterion, 
            device, 
            settings, 
            metric_logger=None, 
            console_logger=None
        ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.settings = settings
        self.training_args = settings.training_args
        self.metric_logger = metric_logger
        self.console_logger = console_logger or logger
        self.accumulated_loss = 0.0

    def _compute_loss(self, input_ids, attention_mask):
        """Standardized Next Token Prediction loss logic."""
        # Shift for causal prediction
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        mask = attention_mask[:, 1:].contiguous().view(-1)

        logits = self.model(inputs)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        loss = self.criterion(logits, labels)
        loss = (loss * mask).sum() / mask.sum()
        
        predictions = torch.argmax(logits, dim=-1)
        correct = ((predictions == labels) * mask).sum().item()
        total = mask.sum().item()
        
        return loss, correct, total

    def train_step(self, input_ids, attention_mask, step_count):
        self.model.train()
        
        # Use Automatic Mixed Precision for speedup
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss, correct, total = self._compute_loss(input_ids, attention_mask)
            scaled_loss = loss / self.training_args.gradient_accumulation_steps
            scaled_loss.backward()
            
        self.accumulated_loss += loss.item()

        if (step_count + 1) % self.training_args.gradient_accumulation_steps == 0:
            # Gradient Norm Logging
            total_norm = self._get_grad_norm()

            clip_grad_norm_(self.model.parameters(), self.training_args.gradient_clipping)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Logging
            accuracy = correct / total
            avg_loss = self.accumulated_loss / self.training_args.gradient_accumulation_steps

            metrics = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'learning_rate': self.scheduler.get_last_lr()[0],
                'grad_norm': total_norm
            }
            if self.metric_logger:
                self.metric_logger.log_metrics(step_count + 1, metrics, self.settings.task, prefix="train")
            
            self.accumulated_loss = 0.0
            return avg_loss, accuracy
        
        return None, None

    @torch.no_grad()
    def evaluate(self, val_loader, max_steps: int, train_step_count: int):
        self.model.eval()
        total_loss, total_correct, total_tokens = 0.0, 0, 0

        for i, (input_ids, attention_mask) in enumerate(val_loader):
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                loss, correct, tokens = self._compute_loss(input_ids, attention_mask)
            
            total_loss += (loss.item() * tokens)
            total_correct += correct
            total_tokens += tokens

            if (i + 1) >= max_steps:
                break

        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }

        if self.metric_logger:
                self.metric_logger.log_metrics(train_step_count + 1, metrics, self.settings.task, prefix="val")

        return avg_loss, accuracy

    def _get_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def save(self, dataloader, epoch, step, checkpoint_dir: str=None):
        """Saves the current trainer state to a checkpoint file."""
        task_dir = os.path.join(checkpoint_dir or self.training_args.checkpoint_dir, self.settings.task)
        os.makedirs(task_dir, exist_ok=True)
        checkpoint_path = os.path.join(task_dir, f"epoch{epoch}_step{step}.pth")
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'dataloader_state_dict': dataloader.state_dict(),
            'epoch': epoch,
            'step': step
        }
        
        torch.save(state, checkpoint_path)
        self.console_logger.info(f"Saved checkpoint to: {checkpoint_path}")

    def load(self, dataloader, checkpoint_path):
        """Loads the trainer state from a checkpoint file."""
        self.console_logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        dataloader.load_state_dict(checkpoint['dataloader_state_dict'])
        
        return checkpoint['epoch'], checkpoint['step']
