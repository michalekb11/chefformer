from datasets import load_dataset
from model.datasets import ChunkedTextDataset
from transformers import AutoTokenizer
from torchtyping import TensorType
import torch
from torch.utils.data import DataLoader
from config.settings import PreTrainingSettings
from model.model import Chefformer

# Create collate function for data loader
def batch_collator(batch: list[(TensorType['max_context_length'], TensorType['max_context_length'])]):
    """Pass this function to the PyTorch DataLoader to properly format inputs for batch"""
    input_ids = torch.stack([ex[0] for ex in batch], dim=0)
    attention_mask = torch.stack([ex[1] for ex in batch], dim=0)
    return input_ids, attention_mask

# Device to use for training
if torch.backends.mps.is_available():  # MPS for Apple Silicon (Mac)
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
else:  # Fallback to CPU
    device = torch.device("cpu")
    print("Using device: CPU")

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# C4 English only dataset
c4 = load_dataset("allenai/c4", "en", streaming=True, split='train')
c4 = ChunkedTextDataset(c4, tokenizer=tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=1000)

# Set up dataloader for training
dataloader = DataLoader(c4, batch_size=PreTrainingSettings.batch_size, collate_fn=batch_collator)

# Optimizer, loss, and model
model = Chefformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=PreTrainingSettings.learning_rate, 
                              weight_decay=PreTrainingSettings.weight_decay)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(PreTrainingSettings.num_epochs):
    for i, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        # We cannot predict for the last token since we don't know what comes after it
        # We need to predict the next token given the sequence that comes before it
        labels = input_ids[:, 1:].contiguous() # (batch_size, seq_len - 1)
        input_ids = input_ids[:, :-1].contiguous() # (batch_size, seq_len - 1)

        # The logits are returned as size (batch_size, seq_len, vocab_size). This needs to be reshaped to (batch_size * seq_len, vocab_size) to compute loss.
        # The labels will also need to be reshaped to (batch_size * seq_len, )
        logits = model(input_ids)
        logits = logits.view(-1, logits.size(-1)) # (batch_size * seq_len, vocab_size)
        labels = labels.view(-1) # (batch_size * seq_len)
        
        # Compute loss
        loss = criterion(logits, labels)

        # Update weights
        loss.backward() 
        optimizer.step()  
        optimizer.zero_grad()  

        # Optionally, track loss for logging purposes
        if i % 100 == 0:  # Every 100 iterations
            print(f"Epoch {epoch+1}, Iteration {i}, Loss: {loss.item()}")

        