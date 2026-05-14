import torch
import torch.nn as nn

def compute_loss(input_ids, attention_mask):
    """Standardized Next Token Prediction loss logic."""
    criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
    
    # Dummy model to simulate Chefformer output
    class DummyModel(nn.Module):
        def __init__(self, vocab_size, embedding_size):
            super().__init__()
            self.linear = nn.Linear(embedding_size, vocab_size)
            # Simulate an embedding layer to get the right input dimension for linear
            self.dummy_embedding = nn.Embedding(vocab_size, embedding_size) 

        def forward(self, x):
            # x is (batch_size, seq_len)
            # Simulate embedding output (batch_size, seq_len, embedding_size)
            dummy_embedded_input = self.dummy_embedding(x % self.dummy_embedding.num_embeddings) 
            # Simulate Chefformer's final linear layer output
            return self.linear(dummy_embedded_input) # (batch_size, seq_len, vocab_size)

    vocab_size = 50257 # GPT-2 vocab size
    embedding_size = 768 # GPT-2 small embedding size
    model = DummyModel(vocab_size, embedding_size)
    
    print(f"Initial input_ids type: {input_ids.dtype}, shape: {input_ids.shape}")
    print(f"Initial attention_mask type: {attention_mask.dtype}, shape: {attention_mask.shape}\n")

    labels = input_ids[:, 1:].clone()
    inputs = input_ids[:, :-1]
    
    print(f"Labels after slicing (input_ids[:, 1:].clone()): type: {labels.dtype}, shape: {labels.shape}")
    print(f"Inputs after slicing (input_ids[:, :-1]): type: {inputs.dtype}, shape: {inputs.shape}\n")

    # Set masked tokens to -100 so CrossEntropyLoss ignores them
    # This replaces your manual 'loss * mask' logic
    labels[attention_mask[:, 1:] == 0] = -100
    
    print(f"Labels after masking (attention_mask[:, 1:] == 0): type: {labels.dtype}, shape: {labels.shape}")
    print(f"Example of masked labels (second row): {labels[1]}\n")

    logits = model(inputs)
    print(f"Logits from model(inputs): type: {logits.dtype}, shape: {logits.shape}\n")

    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    print(labels)
    
    print(f"Logits after view(-1, logits.size(-1)): type: {logits.dtype}, shape: {logits.shape}")
    print(f"Labels after view(-1): type: {labels.dtype}, shape: {labels.shape}\n")

    total_loss_sum = criterion(logits, labels)
    print(f"total_loss_sum from criterion(logits, labels): type: {total_loss_sum.dtype}, shape: {total_loss_sum.shape}\n")
    print(total_loss_sum)

    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        print(predictions)
        mask = (labels != -100)
        print(mask)
        correct = (predictions == labels).sum() # labels is already -100 where masked
        total = mask.sum()
        
        print(f"Predictions (torch.argmax(logits, dim=-1)): type: {predictions.dtype}, shape: {predictions.shape}")
        print(f"Mask (labels != -100): type: {mask.dtype}, shape: {mask.shape}")
        print(f"Correct ((predictions == labels).sum()): type: {correct.dtype}, shape: {correct.shape}")
        print(f"Total (mask.sum()): type: {total.dtype}, shape: {total.shape}\n")
        print(f"Example of predictions (first 10): {predictions[:10]}")
        print(f"Example of labels (first 10): {labels[:10]}")
    
    return total_loss_sum, correct, total

if __name__ == '__main__':
    # Dummy values for input_ids and attention_mask
    batch_size = 2
    seq_len = 10
    
    # Simulate input_ids with some token IDs
    dummy_input_ids = torch.tensor([
        [101, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        [101, 102, 103, 104, 105, 0, 0, 0, 0, 0] # Second sequence has padding
    ], dtype=torch.long)
    
    # Simulate attention_mask where 0 indicates padding
    dummy_attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    ], dtype=torch.long)
    
    total_loss_sum, correct, total = compute_loss(dummy_input_ids, dummy_attention_mask)
    print(f"Final Loss (sum of unreduced losses for valid tokens): {total_loss_sum.item()}")
    print(f"Final Correct predictions: {correct.item()}")
    print(f"Final Total valid tokens: {total.item()}")
    print(f"Final Accuracy: {correct.item() / total.item() if total.item() > 0 else 0.0}")