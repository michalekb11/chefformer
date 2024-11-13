from typing import Optional
import torch
from torch import nn
from transformers import AutoTokenizer
from config.ModelSettings import ModelSettings
from torchtyping import TensorType


# Note: how do we deal with (ignore or remove) the padding tokens? 
# Right now, they are included in the attention output since all sequences must be the same length
# We would like to take the attention output for sequences < max_seq_len, remove the rows corresponding to the padding tokens...
def scaled_dot_product_attention(query: TensorType['batch_size', 'seq_length', 'attn_head_dim'], 
                                 key: TensorType['batch_size', 'seq_length', 'attn_head_dim'], 
                                 value: TensorType['batch_size', 'seq_length', 'attn_head_dim']):
    batch_size, seq_length, d_attn_head = query.shape[0], query.shape[1], query.shape[-1]
    attenion_weights = torch.bmm(query, torch.transpose(key, 1, 2)) # Q * K.T... (batch_size, seq_len, seq_len)
    attenion_weights = attenion_weights / torch.sqrt(torch.tensor(d_attn_head, dtype=torch.float32)) # Scale the scores

    mask = torch.tril(torch.ones(batch_size, seq_length, seq_length, dtype=torch.int)) # Gather attention matrix... (batch_size, seq_len, seq_len)
    attenion_weights[mask == 0] = float('-inf') # Set scores to -inf where tokens are masked

    attenion_weights = torch.softmax(attenion_weights, dim=-1) # Take softmax along rows (each token's row should sum to 1)
    return torch.bmm(attenion_weights, value) # Value multiplication with scores... (batch_size, seq_len, attn_head_output_dim)


class Embeddings(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(model_settings.vocab_size, model_settings.embedding_size)
        self.position_embeddings = nn.Embedding(model_settings.max_context_length, model_settings.embedding_size)

    def forward(self, input_ids: TensorType['batch_size', 'seq_len']) -> TensorType['batch_size', 'seq_len', 'hidden_dim']: # input_ids shape (batch size, seq length (with padding)) 
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand(input_ids.shape[0], -1) # (batch size, seq length)
        embeddings = self.token_embeddings(input_ids) # (batch size, seq_length, hidden dim)
        embeddings = embeddings + self.position_embeddings(position_ids) # (batch size, seq_length, hidden dim)
        return embeddings


class AttentionHead(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        self.q = nn.Linear(in_features=model_settings.embedding_size, 
                           out_features=model_settings.embedding_size // model_settings.num_attn_heads,
                           bias=True)
        self.k = nn.Linear(in_features=model_settings.embedding_size, 
                           out_features=model_settings.embedding_size // model_settings.num_attn_heads,
                           bias=True)
        self.v = nn.Linear(in_features=model_settings.embedding_size, 
                           out_features=model_settings.embedding_size // model_settings.num_attn_heads,
                           bias=True) # After each projection shape will be (batch_size, seq_len, attn_input_dim) where attn_input_dim = hidden_dim // n_attn_heads

    def forward(self, hidden_state: TensorType['batch_size', 'seq_len', 'hidden_dim']):
        return scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
    

class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        self.attn_heads = nn.ModuleList([AttentionHead(model_settings) for _ in range(model_settings.num_attn_heads)])
        self.output_projection = nn.Linear(model_settings.embedding_size, model_settings.embedding_size, bias=True)
        
    def forward(self, hidden_state: TensorType['batch_size', 'seq_len', 'hidden_dim']):
        attn_outputs = torch.cat([head(hidden_state) for head in self.attn_heads], dim=-1)
        return self.output_projection(attn_outputs)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        self.f1 = nn.Linear(model_settings.embedding_size, 4 * model_settings.embedding_size) # To higher dimension (could specify an intermediate size in settings if preferred)
        self.f2 = nn.Linear(4 * model_settings.embedding_size, model_settings.embedding_size) # Back to original dimension
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(model_settings.dropout_prob)

    def forward(self, hidden_state: TensorType['batch_size', 'seq_len', 'hidden_dim']):
        x = self.f1(hidden_state)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.f2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        self.MultiHeadMaskedSelfAttention = MultiHeadMaskedSelfAttention(model_settings)
        self.PositionWiseFeedForward = PositionWiseFeedForward(model_settings)
        #self.layer_norm = nn.LayerNorm(______)
        self.dropout = nn.Dropout(model_settings.dropout_prob)

    def forward(self, hidden_state: TensorType['batch_size', 'seq_len', 'hidden_dim']):
        pass


class Chefformer(nn.Module):
    def __init__(self, tokenizer: AutoTokenizer, model_settings: ModelSettings = ModelSettings()):
        super().__init__()
        self.tokenizer: AutoTokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.Embeddings = Embeddings(model_settings)
        self.DecoderBlocks = nn.ModuleList([DecoderBlock(model_settings) for _ in range(model_settings.num_layers)])
        

        # Only used for testing purposes
        self.AttentionHead = AttentionHead(model_settings)
        self.MultiHeadMaskedSelfAttention = MultiHeadMaskedSelfAttention(model_settings)
        self.PositionWiseFeedForward = PositionWiseFeedForward(model_settings)

    def forward(self, input: list[str]):
        encodings = self.tokenizer(input, return_tensors='pt', padding=True, truncation=True)
        embeddings = self.Embeddings(encodings.input_ids)
        # Add decoder blocks here...


    # Helper functions to test components along the way
    def get_embeddings(self, input: list[str]):
        encodings = self.tokenizer(input, return_tensors='pt', padding=True, truncation=True)
        embeddings = self.Embeddings(encodings.input_ids)
        return embeddings
    
    def test_attention_head(self, input: list[str]):
        embeddings = self.get_embeddings(input)
        attn_output = self.AttentionHead(embeddings)
        return attn_output
    
    def test_multihead_masked_self_attention(self, input: list[str]):
        embeddings = self.get_embeddings(input)
        attn_output = self.MultiHeadMaskedSelfAttention(embeddings)
        return attn_output
    
    def test_feed_forward(self, input: list[str]):
        embeddings = self.get_embeddings(input)
        attn_output = self.MultiHeadMaskedSelfAttention(embeddings)
        ff_output = self.PositionWiseFeedForward(attn_output)
        return ff_output


    
