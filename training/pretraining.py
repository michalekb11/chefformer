from datasets import load_dataset
from model.datasets import ChunkedTextDataset
from transformers import AutoTokenizer
from torchtyping import TensorType
import torch
#from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from config.settings import PreTrainingSettings
from model.model import Chefformer
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import os

#=======Helper functions=======
# Save a checkpoint
def save_checkpoint(model, optimizer, scheduler, dataloader, epoch, step, checkpoint_path="./checkpoints"):
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'dataloader_state_dict': dataloader.state_dict(),
        'epoch': epoch,
        'step': step
    }, os.path.join(checkpoint_path, f"checkpoint_epoch{epoch}_step{step}.pt"))

# Load a checkpoint
def load_checkpoint(model, optimizer, scheduler, dataloader, checkpoint_path):
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
    
# model.apply(init_weights)
#==============
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
c4 = ChunkedTextDataset(c4, tokenizer=tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=10000)

# Set up dataloader for training
dataloader = StatefulDataLoader(c4, 
                                batch_size=PreTrainingSettings.batch_size, 
                                collate_fn=batch_collator,
                                snapshot_every_n_steps=PreTrainingSettings.save_checkpoint_every)

# Optimizer, loss, and model
model = Chefformer().to(device, dtype=torch.float32)
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=PreTrainingSettings.learning_rate, 
                              weight_decay=PreTrainingSettings.weight_decay)
scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Load last checkpoint or start completely fresh
# Resume or start fresh
start_epoch, start_step = 0, 0
checkpoint_path = "./checkpoints/latest_checkpoint.pt" # Set to whatever last checkpoint is... or pass in last checkpoint as command line argument.
if os.path.exists(checkpoint_path):
    start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, dataloader, checkpoint_path)

# For gradient accumulation
accumulated_loss = 0.0

for epoch in range(PreTrainingSettings.num_epochs): # start_epoch, 
    for step, (input_ids, attention_mask) in enumerate(dataloader): # , start=start_step.... don't know if I need this if I use StatefulDataLoader()
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        # We cannot predict for the last token since we don't know what comes after it
        # We need to predict the next token given the sequence that comes before it
        labels = input_ids[:, 1:].contiguous() # (batch_size, seq_len - 1)
        input_ids = input_ids[:, :-1].contiguous() # (batch_size, seq_len - 1)
        attention_mask = attention_mask[:, 1:].contiguous().view(-1)  # Reshape to match loss shape (batch_size * seq_len-1)

        # The logits are returned as size (batch_size, seq_len, vocab_size). This needs to be reshaped to (batch_size * seq_len, vocab_size) to compute loss.
        # The labels will also need to be reshaped to (batch_size * seq_len, )
        logits = model(input_ids)
        logits = logits.view(-1, logits.size(-1)) # (batch_size * seq_len, vocab_size)
        labels = labels.view(-1) # (batch_size * seq_len)
        
        # Compute loss
        loss = criterion(logits, labels)
        loss = loss * attention_mask  # Mask invalid padding tokens
        loss = loss.sum() / attention_mask.sum()  # Normalize by valid token count
        accumulated_loss += loss.item()  # Track accumulated loss for logging

        # Scale loss by accumulation steps and backpropagate
        loss = loss / PreTrainingSettings.gradient_accumulation_steps
        loss.backward() # Update gradients

        if (step + 1) % PreTrainingSettings.gradient_accumulation_steps == 0:
            # Update weights
            clip_grad_norm_(model.parameters(), PreTrainingSettings.gradient_clipping) # clip gradient values
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            #======Additional logging information========
            predictions = torch.argmax(logits, dim=-1)  # Get the predicted token IDs
            correct_tokens = (predictions == labels) * attention_mask  # Mask padding
            accuracy = correct_tokens.sum() / attention_mask.sum()  # Normalize by valid tokens

            # Optionally, track loss for logging purposes
            #if i % 100 == 0:  # Every 100 iterations
            print(f"Epoch {epoch+1}, Step {step + 1}, lr: {round(scheduler.get_last_lr()[0], 6)}, Accuracy: {round(accuracy.item(), 4)}, Loss (avg): {round(accumulated_loss / PreTrainingSettings.gradient_accumulation_steps, 4)}")

            accumulated_loss = 0.0

            if step + 1 % PreTrainingSettings.save_checkpoint_every == 0:
                save_checkpoint(model, optimizer, scheduler, dataloader, epoch, step+1)

# Final gradient update if steps are not divisible by gradient_accumulation_steps
if (step + 1) % PreTrainingSettings.gradient_accumulation_steps != 0:
    clip_grad_norm_(model.parameters(), PreTrainingSettings.gradient_clipping)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()