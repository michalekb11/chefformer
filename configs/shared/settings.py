from pydantic_settings import BaseSettings
import os
import re

def get_latest_checkpoint(checkpoint_dir: str, task: str) -> str | None:
    """Helper to find the most recent checkpoint based on epoch and step."""
    target_dir = os.path.join(checkpoint_dir, task)
    if not os.path.exists(target_dir):
        return None

    # Pattern to capture epoch and step: epoch{E}_step{S}.pth
    pattern = re.compile(r"epoch(\d+)_step(\d+)\.pth")
    
    checkpoints = []
    for filename in os.listdir(target_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            checkpoints.append((epoch, step, filename))

    if not checkpoints:
        return None

    # Sort descending by epoch, then step
    checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return os.path.join(target_dir, checkpoints[0][2])

class ModelSettings(BaseSettings):
    name: str = "Chefformer"
    max_context_length: int = 512
    embedding_size: int = 768
    num_layers: int = 12
    num_attn_heads: int = 12
    dropout_prob: float = 0.1
    vocab_size: int = 50257
    gradient_checkpointing: bool = False