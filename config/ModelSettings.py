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
    