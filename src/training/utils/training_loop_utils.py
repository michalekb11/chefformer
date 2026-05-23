import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

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
    
def batch_collator(batch):
    """
    Collate function with bucketed dynamic padding to optimize MPS performance 
    and reduce memory footprint.
    """
    # Standard GPT-2/MPS defaults if not provided by tokenizer
    PAD_ID = 50256 
    IGNORE_INDEX = -100
    BUCKETS = [128, 256, 384, 512]

    if isinstance(batch[0], dict):
        input_ids = [ex['input_ids'] for ex in batch]
        attention_mask = [ex['attention_mask'] for ex in batch]
        labels = [ex['labels'] for ex in batch]

        max_batch_len = max(len(ids) for ids in input_ids)
        
        # Find the smallest bucket that fits the max length, capping at the largest bucket
        target_len = next((b for b in BUCKETS if max_batch_len <= b), BUCKETS[-1])

        # Efficiently pad to max length in batch and stack into [batch_size, max_batch_len]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Apply a single vectorized padding operation to reach the bucket target length
        pad_len = target_len - input_ids.size(1)
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=PAD_ID)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            labels = F.pad(labels, (0, pad_len), value=IGNORE_INDEX)

        return input_ids, attention_mask, labels
    
    else:
        # Pretraining chunks are usually pre-aligned, 
        # but we ensure batching consistency here.
        input_ids = [ex[0] for ex in batch]
        attention_mask = [ex[1] for ex in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return input_ids, attention_mask, None

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