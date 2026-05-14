import torch
import torch.nn as nn
import torch.nn.functional as F

def trace_attention():
    # Configuration
    batch_size = 2
    seq_len = 4
    embedding_size = 128
    num_heads = 8
    head_dim = embedding_size // num_heads # 16

    print(f"--- Configuration ---")
    print(f"Batch: {batch_size}, SeqLen: {seq_len}, Embed: {embedding_size}, Heads: {num_heads}, HeadDim: {head_dim}\n")

    # 1. Input Tensor
    x = torch.randn(batch_size, seq_len, embedding_size)
    print(f"1. Input Shape (B, S, E): {x.shape}")

    # 2. Linear Projection
    # We use 3 * embedding_size to get Q, K, and V in one go
    qkv_projection = nn.Linear(embedding_size, 3 * embedding_size)
    qkv = qkv_projection(x)
    print(f"2. QKV Projected (B, S, 3*E): {qkv.shape}")

    # 3. Reshape to separate 3 types and H heads
    # We split 3*E into [3, num_heads, head_dim]
    qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
    print(f"3. Reshaped (B, S, 3, H, D): {qkv.shape}")

    # 4. Permute to get Batch and Head to the front
    # Standard format for attention is (B, H, S, D)
    # We want index 2 (the 3 QKV types) to be first so we can unpack
    qkv = qkv.permute(2, 0, 3, 1, 4) 
    print(f"4. Permuted (3, B, H, S, D): {qkv.shape}")

    # 5. Split into Q, K, V
    q, k, v = qkv[0], qkv[1], qkv[2]
    print(f"5. Split Q shape (B, H, S, D): {q.shape}")

    # 6. Optimized Attention
    # This handles the (S x S) matrix math for every head in parallel
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"6. Attention Output (B, H, S, D): {attn_output.shape}")

    # 7. Prepare for Merge
    # We need to move H back next to D so we can concatenate them
    # This makes the tensor NON-CONTIGUOUS
    attn_output = attn_output.transpose(1, 2)
    print(f"7. Transposed (B, S, H, D). Is contiguous? {attn_output.is_contiguous()}")

    # 8. Contiguous and Reshape (The Merge)
    # We merge H and D back into E
    # Using .reshape() is often cleaner than .contiguous().view()
    merged = attn_output.reshape(batch_size, seq_len, embedding_size)
    print(f"8. Final Merged Shape (B, S, E): {merged.shape}")

    # 9. Output Projection
    out_proj = nn.Linear(embedding_size, embedding_size)
    final_output = out_proj(merged)
    print(f"9. Final Linear Projection: {final_output.shape}")

if __name__ == "__main__":
    trace_attention()