## Notes for training the decoder model

1. Handling padding tokens in loss calculation
    - Need to utilize the attention_mask ouput from the tokenizer to apply a Loss Mask. Do not allow loss computation for padding tokens.

2. Needed to create custom Iterable dataset class to stream data and process them into batches correctly.
    - Included chunking to max context length and small overlapping of chunks.

3. Look into PyTorch autocast for floating point numbers if having issues with memory.

4. If the batch size is too small and inefficient, consider gradient accumulation (call loss.backward() multiple times before updating parameters with the optimizer).