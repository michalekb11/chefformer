from dataclasses import dataclass
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

@dataclass
class ModelSettings:
    name: str = config['model']['name']
    vocab_size: int = config['model']['vocab_size']
    embedding_size: int = config['model']['embedding_size']
    max_context_length: int = config['model']['max_context_length']
    num_attn_heads: int = config['model']['num_attn_heads']
    dropout_prob: float = config['model']['dropout_prob']
    num_layers: int = config['model']['num_layers']

@dataclass
class PreTrainingSettings:
    batch_size: int = config['training']['batch_size']
    learning_rate: float = config['training']['learning_rate']
    num_epochs: int = config['training']['num_epochs']
    weight_decay: float = config['training']['weight_decay']
    gradient_clipping: float = config['training']['gradient_clipping']
    warmup_iters: int = config['training']['warmup_iters']
    save_checkpoint_every: int = config['training']['save_checkpoint_every']
    checkpoint_dir: str = config['training']['checkpoint_dir']
    decay_start_iter: int = config['training']['decay_start_iter']
    decay_total_iters: int = config['training']['decay_total_iters']
    gradient_accumulation_steps: int = config['training']['gradient_accumulation_steps']
    