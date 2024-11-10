from dataclasses import dataclass
import os

@dataclass
class ModelSettings:
    vocab_size: int = int(os.getenv('VOCAB_SIZE'))
    embedding_size: int = int(os.getenv('EMBEDDING_SIZE'))
    