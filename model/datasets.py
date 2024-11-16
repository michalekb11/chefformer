import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

# TEMPLATE
# class CustomIterableDataset(IterableDataset):
#     def __init__(self, data):
#         self.data = data

#     def __iter__(self):
#         for item in self.data:
#             yield item

class ChunkedTextDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, tokenizer: AutoTokenizer, chunk_size: int, overlap_size:int=0, shuffle:bool=False, shuffle_buffer_size:int=10000):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        if shuffle:
            self.dataset = self.dataset.shuffle(seed=42, buffer_size=shuffle_buffer_size)


    def __iter__(self):
        token_id_buffer, attn_mask_buffer = [], []
        for example in self.dataset:
            tokenized_example = self.tokenizer(example["text"] + self.tokenizer.eos_token, padding=False, truncation=True) # keep as list for now; also, padding not necessary since all sequences yielded will be length 512

            # Accumulate data in buffers
            token_id_buffer.extend(tokenized_example["input_ids"])
            attn_mask_buffer.extend(tokenized_example["attention_mask"])
            
            # Tokenize in chunks of chunk_size
            while len(token_id_buffer) >= self.chunk_size:
               # Extract a chunk
                chunk_ids = token_id_buffer[:self.chunk_size]
                chunk_attention = attn_mask_buffer[:self.chunk_size]
                
                # Remove used portion from the buffer
                token_id_buffer = token_id_buffer[self.chunk_size - self.overlap_size:]
                attn_mask_buffer = attn_mask_buffer[self.chunk_size - self.overlap_size:]
                
                # Convert to tensors and yield
                yield torch.tensor(chunk_ids, dtype=torch.long), torch.tensor(chunk_attention, dtype=torch.long)
