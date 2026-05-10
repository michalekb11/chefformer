import torch
from torch.nn.utils import clip_grad_norm_
from src.training.utils.training_loop_utils import (
    save_checkpoint, 
    save_training_loss, 
    save_validation_loss, 
    save_learning_rate, 
    save_gradient_norm
)
from loggers.composite_logger import CompositeMetricLogger

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, settings):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.settings = settings
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

    def train_step(self, input_ids, attention_mask, step_count, metric_logger: CompositeMetricLogger):
        self.model.train()
        loss, correct, total = self._compute_loss(input_ids, attention_mask)
        
        # Gradient Accumulation
        scaled_loss = loss / self.settings.gradient_accumulation_steps
        scaled_loss.backward()
        self.accumulated_loss += loss.item()

        if (step_count + 1) % self.settings.gradient_accumulation_steps == 0:
            # Gradient Norm Logging
            total_norm = self._get_grad_norm()

            clip_grad_norm_(self.model.parameters(), self.settings.gradient_clipping)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Logging
            accuracy = correct / total
            avg_loss = self.accumulated_loss / self.settings.gradient_accumulation_steps

            metrics = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'learning_rate': self.scheduler.get_last_lr()[0],
                'grad_norm': total_norm
            }
            metric_logger.log_metrics(step_count + 1, metrics, self.settings.task, prefix="train")
            
            self.accumulated_loss = 0.0
            return avg_loss, accuracy
        
        return None, None

    @torch.no_grad()
    def evaluate(self, val_loader, max_steps):
        self.model.eval()
        total_loss, total_correct, total_tokens = 0.0, 0, 0

        for i, (input_ids, attention_mask) in enumerate(val_loader):
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            loss, correct, tokens = self._compute_loss(input_ids, attention_mask)
            
            total_loss += (loss.item() * tokens)
            total_correct += correct
            total_tokens += tokens

            if (i + 1) >= max_steps:
                break

        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        return avg_loss, accuracy

    def _get_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def save(self, dataloader, epoch, step, path=None):
        save_checkpoint(self.model, self.optimizer, self.scheduler, dataloader, epoch, step)
