from typing import Optional
import torch
from torch import nn
from transformers import AutoTokenizer
from config.ModelSettings import ModelSettings



class Embeddings(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(model_settings.vocab_size, model_settings.embedding_size)
        self.position_embeddings = nn.Embedding(model_settings.max_context_length, model_settings.embedding_size)

    def forward(self, input_ids):
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long)
        embeddings = self.token_embeddings(input_ids)
        embeddings = embeddings + self.position_embeddings(position_ids)
        return embeddings
        



# Anticipated classes
class MultiHeadSelfAttention():
    pass

class DecoderBlock(nn.Module):
    pass

class Chefformer(nn.Module):
    def __init__(self, tokenizer: AutoTokenizer, model_settings: ModelSettings = ModelSettings()):
        super().__init__()
        self.tokenizer: AutoTokenizer = tokenizer
        self.Embeddings = Embeddings(model_settings)


    def generate(self, input: str):
        encodings = self.tokenizer(input, return_tensors='pt')
        embeddings = self.Embeddings(encodings.input_ids)
        
        return embeddings
    

    