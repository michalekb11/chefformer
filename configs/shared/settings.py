from pydantic_settings import BaseSettings

class ModelSettings(BaseSettings):
    name: str = "Chefformer"
    max_context_length: int = 512
    embedding_size: int = 768
    num_layers: int = 12
    num_attn_heads: int = 12
    dropout_prob: float = 0.1
    vocab_size: int = 50257