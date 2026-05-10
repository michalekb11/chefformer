import torch
from torchtyping import TensorType

# Define warmup and decay schedule
def lr_schedule(step, warmup_iters, decay_start_iter, decay_total_iters):
    """Should return a multiplier for the learning rate"""
    if step < warmup_iters:
        return step / warmup_iters  # Linear warmup
    elif step >= decay_start_iter:
        progress = torch.tensor((step - decay_start_iter), dtype=torch.float32) / decay_total_iters
        return max(0.0, 0.5 * (1.0 + torch.cos(progress * torch.pi)).item())  # Cosine decay
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