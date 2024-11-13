## Transformer architecture

### Pieces
0. Large pretraining dataset
    - Creating/using HuggingFace dataset makes it easier to use large amounts of data on a small machine (through memory mapping and streaming). 
    - Essentially, it uses a pointer to a place on disk (this acts as an extension of RAM).
    - "DownloadConfig(delete_extracted=True)". Then load_dataset().
    - If you can't free enough disk space to store the full dataset locally, use streaming (load_dataset(streaming=True)).

1. Tokenizer
    - For now, we can just use a HuggingFace pretrained tokenizer until we are ready to train our own tokenizer.
    - For ease of use, lets use tokenizer = AutoTokenizer.from_pretrained("gpt2") for now.

2. Embeddings
    - Utilize nn.Embeddings to create an n-dimensional vector for each input token?
    - This represents a lookup table for each possible token of dictionary. Then, we also must specify the hidden dimension size that each embedding should be.
    - Transforming to embeddings results in the shape [batch_size, n_tokens, hidden_size].

3. Positional Embeddings
    - Need to decide on a max context length. 512 is a good happy medium, but I'm not sure how many recipes fit into this context length. Could also choose 256 or 1024, etc.
    - We do not have to use the same context length as GPT-2 (the tokenizer) since we are training our own model.
    - Typical strategies include learned positional embeddings and non-learned ones. Non-learned could be utilizing a sinusoidal function to encode the position of each token.
    - Here, we will try a learned method by allowing for an embedding vector to be created for each possible position (up to the max context length).
    
4. Attention Mechanism
    - We need casual attention (masked self-attention) since the decoder can only attend to tokens that come before the current token in the sequence.
    - Scaled dot product attention is the most common attention.
    - D_attn for the attention weight matrix does not necessarily have to be same as embedding dimension. It can be smaller to control computational cost.
    - D_attn is split up among the heads. If it is 512 / 8 heads = 64 per head.
    
5. Multi-Head Attention
    - x
    
6. Feedforward Network
    - x
    
7. Layer Normalization and Residual Connections
    - x
    
8. Single Decoder Block
    - x
    
9. Stack Decoder Blocks
    - x
    
10. Final Linear and Softmax Layers
    - x
    

NOTE: Use MPS device for PyTorch (GPU training for macOS).