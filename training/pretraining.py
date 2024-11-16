from datasets import load_dataset
from model.datasets import ChunkedTextDataset
from transformers import AutoTokenizer
from torchtyping import TensorType
import torch
from torch.utils.data import DataLoader
from config.settings import PreTrainingSettings

# Create collate function for data loader
def batch_collator(batch: list[(TensorType['max_context_length'], TensorType['max_context_length'])]):
    """Pass this function to the PyTorch DataLoader to properly format inputs for batch"""
    input_ids = torch.stack([ex[0] for ex in batch], dim=0)
    attention_mask = torch.stack([ex[1] for ex in batch], dim=0)
    return input_ids, attention_mask

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# C4 English only dataset
c4 = load_dataset("allenai/c4", "en", streaming=True, split='train')
c4 = ChunkedTextDataset(c4, tokenizer=tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=1000)

# Set up dataloader for training
dataloader = DataLoader(c4, batch_size=PreTrainingSettings.batch_size, collate_fn=batch_collator)

for i, (input_ids, attention_mask) in enumerate(dataloader):
    print(f"Batch {i}")
    print("Input IDs shape:", input_ids.shape)  # Example: torch.Size([32, 512]) for a batch of 32 sequences of length 512
    print("Attention Mask shape:", attention_mask.shape)
    print()
    if i == 3:
        break