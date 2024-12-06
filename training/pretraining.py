from datasets import load_dataset
from model.datasets import ChunkedTextDataset
from transformers import AutoTokenizer
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from config.settings import PreTrainingSettings
from model.model import Chefformer
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import os
from helper_functions import save_checkpoint, load_checkpoint, lr_schedule, batch_collator, init_weights

#=======User parameters=======
latest_checkpoint_path = './checkpoints/latest_checkpoint.pth'

#=======Validation loop definition=======
def val_loop(model: Chefformer, criterion: torch.nn.CrossEntropyLoss, device: torch.device):
    # C4 English only dataset (We will only use the first 2000. The seed is set inside ChunkedTextDataset, so we should get the same ones each time.)
    print("Loading C4 validation set...")
    c4_val = load_dataset("allenai/c4", "en", streaming=True, split='validation')
    c4_val = ChunkedTextDataset(c4, tokenizer=tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=PreTrainingSettings.validation_loop_steps + 100)

    # Set up dataloader for training
    dataloader = StatefulDataLoader(c4_val, 
                                    batch_size=PreTrainingSettings.batch_size, 
                                    collate_fn=batch_collator #,
                                    #snapshot_every_n_steps=PreTrainingSettings.save_checkpoint_every
                                    )
    
    # Start the evaluation
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():  # Disable gradient computation
        for cur_validation_step, (input_ids, attention_mask) in enumerate(dataloader):
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
            predictions = torch.argmax(logits, dim=-1)  # Get the predicted token IDs
            
            # Compute loss
            loss = criterion(logits, labels)
            loss = loss * attention_mask  # Mask invalid padding tokens
            loss = loss.sum() #/ attention_mask.sum()  # Normalize by valid token count
            
            total_tokens += attention_mask.sum().item()
            total_loss += loss.item()
            total_correct += ((predictions == labels) * attention_mask).sum().item()

            # Print updates
            if (cur_validation_step + 1) % 10 == 0:
                print(f'Processed validation step: {cur_training_step + 1}')

            # Only do the first 2000 steps
            if (cur_validation_step + 1) == PreTrainingSettings.validation_loop_steps:
                break

    # Compute/log loss metrics
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    print(f"Accuracy: {round(accuracy, 4)}, Validation loss (avg): {round(avg_loss, 4)}") # Accuracy: {round(accuracy.item(), 4)}, Loss (avg):
    return avg_loss, accuracy

#=======Model and dataset setup=======
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
                                collate_fn=batch_collator #,
                                #snapshot_every_n_steps=PreTrainingSettings.save_checkpoint_every
                                )

# Optimizer, loss, and model
model = Chefformer().to(device, dtype=torch.float32)
# model.apply(init_weights) # For Xavier weight initialization
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=PreTrainingSettings.learning_rate, 
                              weight_decay=PreTrainingSettings.weight_decay)
scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Load last checkpoint or start completely fresh
start_epoch, start_step = 0, 0
if os.path.exists(latest_checkpoint_path):
    start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, dataloader, latest_checkpoint_path)

#=======Training loop=======
# For gradient accumulation
accumulated_loss = 0.0

for epoch in range(start_epoch, PreTrainingSettings.num_epochs):
    print(f'Training at epoch: {epoch}')
    print(f'Training at start step: {start_step}')
    
    for cur_training_step, (input_ids, attention_mask) in enumerate(dataloader): # , start=start_step.... don't need this if I use StatefulDataLoader()
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

        if (cur_training_step + 1) % PreTrainingSettings.gradient_accumulation_steps == 0:
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
            print(f"Epoch {epoch+1}, Step {start_step + cur_training_step + 1}, lr: {round(scheduler.get_last_lr()[0], 6)}, Accuracy: {round(accuracy.item(), 4)}, Loss (avg): {round(accumulated_loss / PreTrainingSettings.gradient_accumulation_steps, 4)}")

            accumulated_loss = 0.0

        # Save model every N steps
        if (cur_training_step + 1) % PreTrainingSettings.save_checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, dataloader, epoch, start_step + cur_training_step + 1)

        # Start validation loop every M steps
        if (cur_training_step + 1) % PreTrainingSettings.validate_every == 0:
            avg_loss, accuracy = val_loop(model, device)
            model.train()
        
    # If we reach the end of an epoch, checkpoint the model
    save_checkpoint(model, optimizer, scheduler, dataloader, epoch, start_step + cur_training_step + 1)
    start_step = 0

# Final gradient update and checkpoint if steps are not divisible by gradient_accumulation_steps
if (cur_training_step + 1) % PreTrainingSettings.gradient_accumulation_steps != 0:
    clip_grad_norm_(model.parameters(), PreTrainingSettings.gradient_clipping)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    save_checkpoint(model, optimizer, scheduler, dataloader, epoch, start_step + cur_training_step + 1)

print('Pre-training complete.')