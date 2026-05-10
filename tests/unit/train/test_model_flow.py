import torch
import pytest
from src.shared.model.model import Chefformer
from configs.shared.settings import ModelSettings
from src.shared.model.model import Embeddings, MultiHeadMaskedSelfAttention, PositionWiseFeedForward

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

def test_component_shapes():
    """Verify that individual components return the expected tensor shapes."""
    vocab_size = 500
    hidden_dim = 128
    seq_len = 16
    batch_size = 2
    settings = ModelSettings(
        embedding_size=hidden_dim, 
        vocab_size=vocab_size, 
        max_context_length=64,
        num_attn_heads=4
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 1. Test Embeddings
    emb_module = Embeddings(settings)
    emb_out = emb_module(input_ids)
    assert emb_out.shape == (batch_size, seq_len, hidden_dim)
    
    # 2. Test Attention
    attn_module = MultiHeadMaskedSelfAttention(settings)
    attn_out = attn_module(emb_out)
    assert attn_out.shape == (batch_size, seq_len, hidden_dim)
    
    # 3. Test FeedForward
    ff_module = PositionWiseFeedForward(settings)
    ff_out = ff_module(emb_out)
    assert ff_out.shape == (batch_size, seq_len, hidden_dim)

def test_causal_masking(model):
    """Verify that the model is indeed causal (tokens don't look ahead)."""
    model.eval()
    batch_size, seq_len = 1, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # Pass 1: Get logits for the full sequence
    logits_1 = model(input_ids)

    # Pass 2: Modify ONLY the very last token in the sequence
    input_ids_modified = input_ids.clone()
    input_ids_modified[0, -1] = (input_ids[0, -1] + 1) % 1000
    logits_2 = model(input_ids_modified)

    # The logits for all tokens EXCEPT the modified last one should be identical.
    # If the mask is working, changing index 9 cannot affect the output for indices 0-8.
    torch.testing.assert_close(logits_1[:, :-1, :], logits_2[:, :-1, :], msg="Causal mask failure: Future tokens are affecting the past.")

def test_chefformer_gradient_flow(model):
    """Check if gradients actually propagate back to the embeddings."""
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    
    # Check if embedding weights have gradients
    assert model.Embeddings.token_embeddings.weight.grad is not None