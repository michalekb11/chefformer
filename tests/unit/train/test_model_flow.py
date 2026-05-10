import torch
import pytest
from src.shared.model.model import Chefformer
from configs.shared.settings import ModelSettings

@pytest.fixture
def model():
    settings = ModelSettings(embedding_size=144, num_layers=2, vocab_size=1000)
    return Chefformer(settings)

def test_chefformer_forward_pass(model):
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, 1000)
    assert not torch.isnan(logits).any(), "Model produced NaN outputs"

def test_chefformer_gradient_flow(model):
    """Check if gradients actually propagate back to the embeddings."""
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    
    # Check if embedding weights have gradients
    assert model.Embeddings.token_embeddings.weight.grad is not None