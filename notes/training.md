## Notes for training the decoder model

1. Handling padding tokens in loss calculation
    - Need to utilize the attention_mask ouput from the tokenizer to apply a Loss Mask. Do not allow loss computation for padding tokens.