import os
import torch
from tqdm import tqdm
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

        self.grad_scale_factor = torch.tensor(
            self.training_args.batch_size * (self.settings.model.max_context_length - 1) * self.training_args.gradient_accumulation_steps,
            device=self.device,
            dtype=torch.float32
        )
        
        # Initialize accumulators as tensors on the device to avoid CPU syncs
        self.accumulated_loss = torch.tensor(0.0, device=self.device)
        self.accumulated_correct = torch.tensor(0.0, device=self.device)
        self.accumulated_total = torch.tensor(0.0, device=self.device)

    def _compute_loss(self, input_ids, attention_mask):
        """Standardized Next Token Prediction loss logic."""
        # Shift for causal prediction
        labels = input_ids[:, 1:].clone()
        inputs = input_ids[:, :-1]

        # Set masked tokens to -100 so CrossEntropyLoss ignores them
        # This replaces the manual 'loss * mask' logic
        labels[attention_mask[:, 1:] == 0] = -100

        logits = self.model(inputs)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        loss_sum = self.criterion(logits, labels)

        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            mask = (labels != -100)
            correct = (predictions == labels).sum() # labels is already -100 where masked
            total = mask.sum()
        
        return loss_sum, correct, total

    def train_step(self, input_ids, attention_mask, step_count):
        self.model.train()
        
        # Use Automatic Mixed Precision for speedup
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss_sum, correct, total = self._compute_loss(input_ids, attention_mask)
            # Calculate theoretical max for gradient scaling consistent with global batch size but we use accumulated_total for the actual reported loss metric.
            scaled_loss = loss_sum / self.grad_scale_factor
        
        scaled_loss.backward()
            
        # Accumulate metrics on device
        self.accumulated_loss.add_(loss_sum.detach()) 
        self.accumulated_correct.add_(correct)
        self.accumulated_total.add_(total)

        if (step_count + 1) % self.training_args.gradient_accumulation_steps == 0:
            # Calculate norm before clipping and zeroing to see the "true" gradient magnitude
            total_norm = self._get_grad_norm()

            clip_grad_norm_(self.model.parameters(), self.training_args.gradient_clipping)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Logging
            # Perform a single sync point here for logging
            accuracy = (self.accumulated_correct / self.accumulated_total).item() if self.accumulated_total > 0 else 0.0
            avg_loss = (self.accumulated_loss / self.accumulated_total).item() if self.accumulated_total > 0 else 0.0

            metrics = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'learning_rate': self.scheduler.get_last_lr()[0],
                'grad_norm': total_norm
            }
            if self.metric_logger:
                self.metric_logger.log_metrics(step_count + 1, metrics, self.settings.task, prefix="train")
            
            # Reset accumulators on device
            self.accumulated_loss.zero_()
            self.accumulated_correct.zero_()
            self.accumulated_total.zero_()
            return avg_loss, accuracy
        
        return None, None

    @torch.no_grad()
    def evaluate(self, val_loader, max_steps: int, train_step_count: int):
        self.model.eval()
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0.0, device=self.device)
        total_tokens = torch.tensor(0.0, device=self.device)

        with tqdm(total=max_steps, desc="Evaluating", leave=False) as pbar:
            for i, (input_ids, attention_mask) in enumerate(val_loader):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    loss_sum, correct, tokens = self._compute_loss(input_ids, attention_mask)
                
                # Vectorized accumulation on device
                total_loss.add_(loss_sum)
                total_correct.add_(correct)
                total_tokens.add_(tokens)

                pbar.update(1)
                if (i + 1) >= max_steps:
                    break

        avg_loss = (total_loss / total_tokens).item()
        accuracy = (total_correct / total_tokens).item()

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }

        if self.metric_logger:
            self.metric_logger.log_metrics(train_step_count, metrics, self.settings.task, prefix="val")

        return avg_loss, accuracy

    def _get_grad_norm(self):
        total_norm_sq = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            if p.grad is not None:
                # Compute norm entirely on device to avoid per-parameter syncs
                total_norm_sq += torch.norm(p.grad.detach(), 2) ** 2
        return (total_norm_sq ** 0.5).item()

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
