import os
import torch
from config.settings import PreTrainingSettings
from torchtyping import TensorType
import csv

#=======Helper functions=======
# Save a checkpoint
def save_checkpoint(model, optimizer, scheduler, dataloader, epoch, step, checkpoint_dir=PreTrainingSettings.checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'dataloader_state_dict': dataloader.state_dict(),
        'epoch': epoch,
        'step': step
    }, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")
    return

# Load a checkpoint
def load_checkpoint(model, optimizer, scheduler, dataloader, checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    dataloader.load_state_dict(checkpoint['dataloader_state_dict'])
    return checkpoint['epoch'], checkpoint['step']

# Define warmup and decay schedule
def lr_schedule(step):
    """Should return a multiplier for the learning rate"""
    if step < PreTrainingSettings.warmup_iters:
        return step / PreTrainingSettings.warmup_iters  # Linear warmup
    elif step >= PreTrainingSettings.decay_start_iter:
        progress = (step - PreTrainingSettings.decay_start_iter) / PreTrainingSettings.decay_total_iters
        return max(0.0, 0.5 * (1.0 + torch.cos(progress * torch.pi)))  # Cosine decay
    else:
        return 1.0
    
# Create collate function for data loader
def batch_collator(batch: list[(TensorType['max_context_length'], TensorType['max_context_length'])]):
    """Pass this function to the PyTorch DataLoader to properly format inputs for batch"""
    input_ids = torch.stack([ex[0] for ex in batch], dim=0)
    attention_mask = torch.stack([ex[1] for ex in batch], dim=0)
    return input_ids, attention_mask

def init_weights(m):
    """Apply Xavier weight initialization to all linear layers in the model"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight) # or _normal_()
        m.bias.data.fill_(0.01)

def save_training_loss(training_step: int, training_loss: float, accuracy: float):
    with open('./checkpoints/training_loss.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([training_step, accuracy, training_loss])
    return

def save_validation_loss(training_step: int, accuracy: float, val_loss: float):
    with open('./checkpoints/validation_loss.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([training_step, accuracy, val_loss])
    return

def save_learning_rate(training_step: int, lr: float):
    with open('./checkpoints/learning_rate.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([training_step, lr])
    return

def save_gradient_norm(training_step: int, gradient_norm: float):
    with open('./checkpoints/gradient_norm.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([training_step, gradient_norm])
    return