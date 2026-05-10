from datasets import load_dataset
from src.training.utils.data_utils import ChunkedTextDataset
from transformers import AutoTokenizer
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from src.shared.model.model import Chefformer
from torch.optim.lr_scheduler import LambdaLR
import os
from src.training.utils.training_loop_utils import (
    load_checkpoint, 
    lr_schedule,
    batch_collator, 
    init_weights,
    save_validation_loss
)
from configs.training.settings import pretraining_settings

def run_val_only(pretraining_settings: dict):
    #=======User parameters=======
    latest_checkpoint_path = './checkpoints/checkpoint_epoch0_step14000.pth'

    #=======Validation loop definition=======
    def val_loop(model: Chefformer, criterion: torch.nn.CrossEntropyLoss, device: torch.device):
        # C4 English only dataset (We will only use the first 2000. The seed is set inside ChunkedTextDataset, so we should get the same ones each time.)
        print("Loading C4 validation set...")
        c4_val = load_dataset("allenai/c4", "en", streaming=True, split='validation')
        c4_val = ChunkedTextDataset(c4_val, tokenizer=tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=pretraining_settings.validation_loop_steps + 100)

        # Set up dataloader for training
        dataloader_val = StatefulDataLoader(c4_val, 
                                    batch_size=pretraining_settings.batch_size, 
                                    collate_fn=batch_collator,
                                    #snapshot_every_n_steps=pretraining_settings.save_checkpoint_every,
                                    num_workers=2
                                    )
        
        # Start the evaluation
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():  # Disable gradient computation
            for cur_validation_step, (input_ids, attention_mask) in enumerate(dataloader_val):
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
                    print(f'Processed validation step: {cur_validation_step + 1}, Accuracy: {total_correct / total_tokens}, Loss (avg): {total_loss / total_tokens}')

                # Only do the first 2000 steps
                if (cur_validation_step + 1) == pretraining_settings.validation_loop_steps:
                    break

        # Compute/log loss metrics
        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        print(f"Finished validation loop. Accuracy: {round(accuracy, 4)}, Validation loss (avg): {round(avg_loss, 4)}")
        return avg_loss, accuracy
    
    #=======Model and dataset setup=======
    # Device to use for training
    if torch.backends.mps.is_available():  # MPS for Apple Silicon (Mac)
        device = torch.device("mps")
    else:  # Fallback to CPU
        device = torch.device("cpu")

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # C4 English only dataset
    c4 = load_dataset("allenai/c4", "en", streaming=True, split='train')
    c4 = ChunkedTextDataset(c4, tokenizer=tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=10000)

    # Set up dataloader for training
    dataloader = StatefulDataLoader(c4, 
                                    batch_size=pretraining_settings.batch_size, 
                                    collate_fn=batch_collator,
                                    snapshot_every_n_steps=pretraining_settings.save_checkpoint_every,
                                    num_workers=2
                                    )

    # Optimizer, loss, and model
    model = Chefformer().to(device, dtype=torch.float32)
    model.apply(init_weights) # For Xavier weight initialization
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=pretraining_settings.learning_rate, 
                                weight_decay=pretraining_settings.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Load last checkpoint or start completely fresh
    start_epoch, start_step = 0, 0
    if os.path.exists(latest_checkpoint_path):
        start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, dataloader, latest_checkpoint_path)

    avg_loss, accuracy = val_loop(model, criterion, device)
    save_validation_loss(14000, accuracy, avg_loss)
    model.train()

if __name__ == '__main__':
    run_val_only(pretraining_settings)