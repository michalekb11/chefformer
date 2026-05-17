import torch
import torch.nn as nn
from torchtyping import TensorType

# Define warmup and decay schedule
def lr_schedule(step, warmup_iters, decay_start_iter, decay_total_iters, min_lr_ratio=0.1):
    """Should return a multiplier for the learning rate"""
    if step < warmup_iters:
        return step / warmup_iters  # Linear warmup
    elif step >= decay_start_iter:
        progress = torch.tensor((step - decay_start_iter), dtype=torch.float32) / decay_total_iters
        # HARD CODE MIN LEARNING RATE 10% OF MAX
        return max(min_lr_ratio, 0.5 * (1.0 + torch.cos(progress * torch.pi)).item())  # Cosine decay
    else:
        return 1.0
    
# Create collate function for data loader
def batch_collator(batch: list[(TensorType['max_context_length'], TensorType['max_context_length'])]):
    """Pass this function to the PyTorch DataLoader to properly format inputs for batch"""
    input_ids = torch.stack([ex[0] for ex in batch], dim=0)
    attention_mask = torch.stack([ex[1] for ex in batch], dim=0)
    return input_ids, attention_mask

def init_weights(m):
    """
    Standard Transformer initialization:
    - Linear/Embedding: Normal(0, 0.02)
    - LayerNorm: weight=1, bias=0
    """
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)