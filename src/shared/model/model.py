import torch
from torch import nn
from configs.shared.settings import ModelSettings
from torchtyping import TensorType

#device = torch.device("mps")

########
#self.tokenizer: AutoTokenizer = tokenizer
#self.tokenizer.pad_token = self.tokenizer.eos_token
#x = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True) # Encodings
########

class Embeddings(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(model_settings.vocab_size, model_settings.embedding_size)
        self.position_embeddings = nn.Embedding(model_settings.max_context_length, model_settings.embedding_size)
        self.dropout = nn.Dropout(model_settings.dropout_prob)

    def forward(self, input_ids: TensorType['batch_size', 'seq_len']) -> TensorType['batch_size', 'seq_len', 'hidden_dim']: # input_ids shape (batch size, seq length (with padding)) 
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand(input_ids.shape[0], -1).to(input_ids.device) # (batch size, seq length)
        embeddings = self.token_embeddings(input_ids) # (batch size, seq_length, hidden dim)
        embeddings = embeddings + self.position_embeddings(position_ids) # (batch size, seq_length, hidden dim)
        # Optional layer norm here. Add if seeing gradient stability issues during training.
        embeddings = self.dropout(embeddings)
        return embeddings


class AttentionHead(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        self.head_dim = model_settings.embedding_size // model_settings.num_attn_heads

        self.q = nn.Linear(in_features=model_settings.embedding_size, 
                           out_features=self.head_dim,
                           bias=True)
        self.k = nn.Linear(in_features=model_settings.embedding_size, 
                           out_features=self.head_dim,
                           bias=True)
        self.v = nn.Linear(in_features=model_settings.embedding_size, 
                           out_features=self.head_dim,
                           bias=True) # After each projection shape will be (batch_size, seq_len, attn_input_dim) where attn_input_dim = hidden_dim // n_attn_heads
    
    def scaled_dot_product_attention(self,
                                     query: TensorType['batch_size', 'seq_length', 'attn_head_dim'], 
                                     key: TensorType['batch_size', 'seq_length', 'attn_head_dim'], 
                                     value: TensorType['batch_size', 'seq_length', 'attn_head_dim']):
        batch_size, seq_length, d_attn_head = query.shape[0], query.shape[1], query.shape[-1]
        attenion_weights = torch.bmm(query, torch.transpose(key, 1, 2)) # Q * K.T... (batch_size, seq_len, seq_len)
        attenion_weights = attenion_weights / torch.sqrt(torch.tensor(d_attn_head, dtype=torch.float32)) # Scale the scores

        mask = torch.tril(torch.ones(batch_size, seq_length, seq_length, dtype=torch.int)) # Gather attention matrix... (batch_size, seq_len, seq_len)
        attenion_weights[mask == 0] = float('-inf') # Set scores to -inf where tokens are masked

        attenion_weights = torch.softmax(attenion_weights, dim=-1) # Take softmax along rows (each token's row should sum to 1)
        return torch.bmm(attenion_weights, value) # Value multiplication with scores... (batch_size, seq_len, attn_head_output_dim)
    
    def forward(self, hidden_state: TensorType['batch_size', 'seq_len', 'hidden_dim']):
        return self.scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
    

class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        assert model_settings.embedding_size % model_settings.num_attn_heads == 0, \
            f"embedding_size ({model_settings.embedding_size}) must be divisible by num_attn_heads ({model_settings.num_attn_heads})"
         
        self.attn_heads = nn.ModuleList([AttentionHead(model_settings) for _ in range(model_settings.num_attn_heads)])
        self.output_projection = nn.Linear(model_settings.embedding_size, model_settings.embedding_size, bias=True)
        self.dropout = nn.Dropout(model_settings.dropout_prob)
        
    def forward(self, hidden_state: TensorType['batch_size', 'seq_len', 'hidden_dim']):
        attn_outputs = torch.cat([head(hidden_state) for head in self.attn_heads], dim=-1)
        attn_outputs = self.dropout(attn_outputs)
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
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, model_settings: ModelSettings) -> None:
        super().__init__()
        self.MultiHeadMaskedSelfAttention = MultiHeadMaskedSelfAttention(model_settings)
        self.PositionWiseFeedForward = PositionWiseFeedForward(model_settings)
        self.layer_norm1 = nn.LayerNorm(model_settings.embedding_size)
        self.layer_norm2 = nn.LayerNorm(model_settings.embedding_size)
        self.dropout = nn.Dropout(model_settings.dropout_prob)

    def forward(self, hidden_state: TensorType['batch_size', 'seq_len', 'hidden_dim']):
        x = self.layer_norm1(hidden_state)
        x = self.MultiHeadMaskedSelfAttention(x) + hidden_state # First skip connection
        x = self.dropout(x)
        x = self.PositionWiseFeedForward(self.layer_norm2(x)) + x # Second skip connection
        x = self.dropout(x)
        return x


class Chefformer(nn.Module):
    def __init__(self, model_settings: ModelSettings = ModelSettings()):
        super().__init__()
        self.Embeddings = Embeddings(model_settings)
        self.DecoderBlocks = nn.ModuleList([DecoderBlock(model_settings) for _ in range(model_settings.num_layers)])
        self.layer_norm_final = nn.LayerNorm(model_settings.embedding_size)
        self.unembedding_matrix = nn.Linear(model_settings.embedding_size, model_settings.vocab_size)
        self.n_params = self._count_total_parameters()

    def forward(self, x: TensorType['batch_size', 'seq_len']):
        x = self.Embeddings(x) # Embeddings
        for block in self.DecoderBlocks: # All of the decoder blocks (attention + feed forward)
            x = block(x)
        x = self.layer_norm_final(x) # Final layer norm
        x = self.unembedding_matrix(x) # Unembedding to convert back to (batch_size, seq_len, vocab_size)
        return x
    
    def _count_total_parameters(self):
        total_params = 0
        for p in self.parameters():
            total_params += p.numel()
        print(f"Total parameters: {total_params/1000**2:.1f}M ")
        return total_params
