import torch
from src.shared.model.model import Chefformer
from configs.shared.settings import ModelSettings

def test_weight_update_step():
    """E2E style test: Ensure weights actually change after a backward pass and step."""
    settings = ModelSettings(embedding_size=144, num_layers=1, vocab_size=500)
    model = Chefformer(settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Capture initial weights of a decoder layer
    initial_weights = model.DecoderBlocks[0].layer_norm1.weight.clone().detach()
    
    # Dummy training step
    input_ids = torch.randint(0, 500, (2, 10))
    logits = model(input_ids)
    
    # Target is just shifted input for next-token prediction
    target = torch.randint(0, 500, (2, 10))
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 500), target.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    updated_weights = model.DecoderBlocks[0].layer_norm1.weight.detach()
    
    # Assert that weights have changed
    diff = torch.abs(initial_weights - updated_weights).sum()
    assert diff > 0, "Weights did not update after training step"