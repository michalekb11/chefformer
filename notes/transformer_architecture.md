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
    
5. Multi-Head Attention
    - D_attn is split up among the heads. If it is 512 / 8 heads = 64 per head.
    - The hidden states (to start, the embeddings) are passed to each attention head, for which the output is a projection to a smaller space (D_attn // num_heads).
    - After this, we must concatenate them back together to get the full D_attn. (Question: why does separating them allow them to learn different information? Why cant different info be learned just through 1 head?)
    - A final linear projection is applied after concatenating them back together (so you can project into any other dimension??)
    
6. Feedforward Network
    - Apparently, this is comparable to a one-dimensional convolution with a kernel size of one. This is not the same as a traditional feed forward network (standard linear layers that process the whole sequence of embeddings as 1 vector). This is a position-wise feed forward network, meaning that each embedding is processed independently.
    - This means that the same learned weights are applied to each token in the sequence (the weights are not different for each token).
    - In a standard FFN, if the input layer was flattened, you could in theory combine information across tokens.
    - The position-wise feed-forward layer adds non-linearity and depth to each token representation after the self-attention layers, helping the model learn complex transformations for each token independently. This provides the network with greater capacity for learning without inter-token dependencies in this layer itself. Self-attenion handles the inter-token relationships.
    - Rule of thumb from literature is that the hidden size of the first layer should be at least 4X as big as the embedding dimension.
    - GELU activation is most commonly used. ReLU activation acts as more of an all or nothing gate, allowing input to pass through completely or get blocked entirely. GELU scales inputs more based on the "probability" that they are active, allowing inputs near 0 to have partial activation. ReLU discards some useful information when the inputs are below 0. GELU allows small negative values to pass through which is useful in Transformers since they need to capture small nuanced representations of words or phrases.
    - Typically the two-layer network is composed as follows: Linear layer that projects into higher dimension, GELU, Linear layer that projects back down into original dimension.
    
7. Layer Normalization and Residual Connections
    - x
    
8. Single Decoder Block
    - x
    
9. Stack Decoder Blocks
    - x
    
10. Final Linear and Softmax Layers
    - x
    

NOTE: Use MPS device for PyTorch (GPU training for macOS).
NOTE: Might want to switch to also specifying a attention output size for clarity?