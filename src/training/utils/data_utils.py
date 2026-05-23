import unicodedata
import re
import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from loggers.console_logger import ConsoleLogger

logger = ConsoleLogger(__name__)

class TextCleaner:
    def __init__(self) -> None:
        pass
        
    def remove_accents(self, text):
        # Normalize text to decompose characters (NFD - Normal Form Decomposition)
        normalized_text = unicodedata.normalize('NFD', text)
        # Filter out the accent marks (category "Mn" denotes non-spacing marks)
        ascii_text = ''.join([c for c in normalized_text if not unicodedata.category(c) == 'Mn'])
        # Normalize back to NFC to recombine any remaining composed characters
        return unicodedata.normalize('NFC', ascii_text)
    
    def remove_advertisement_str(self, text):
        return text.replace('ADVERTISEMENT', '').strip()
    
    def replace_whitespace(self, text):
        replace_dict = {
            '\n':'',
            '\t':''
        }
        for k, v in replace_dict.items():
            text = text.replace(k, v)

        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def replace_phrases(self, text):
        replace_dict = {
            'Watch how to make this recipe.':''
        }
        for k, v in replace_dict.items():
            text = text.replace(k, v)
        return text
    
    def remove_html_tags(self, text):
        # Use regex to find any HTML tags and replace them with an empty string
        return re.sub(r'<[^>]+>.*?</[^>]+>', '', text)
    
    def remove_special_characters(self, text):
        return re.sub(r"[^a-zA-Z0-9 .,/:;\'\"[\]{}+=\-_–()*&^%$#@!~\\|°<>?]", '', text)
    

class ChunkedTextDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, tokenizer: AutoTokenizer, chunk_size: int, overlap_size:int=0, shuffle:bool=False, shuffle_buffer_size:int=10000):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        # State variables for checkpointing
        self.num_examples_seen = 0
        self.token_id_buffer = []
        self.attn_mask_buffer = []

        if shuffle:
            self.dataset = self.dataset.shuffle(seed=42, buffer_size=shuffle_buffer_size)
        
    def state_dict(self):
        """Returns the current state of the dataset for checkpointing."""
        return {
            "num_examples_seen": self.num_examples_seen,
            "token_id_buffer": self.token_id_buffer,
            "attn_mask_buffer": self.attn_mask_buffer,
        }

    def load_state_dict(self, state_dict):
        """Restores the dataset state from a checkpoint."""
        if state_dict:
            self.num_examples_seen = state_dict.get("num_examples_seen", 0)
            self.token_id_buffer = state_dict.get("token_id_buffer", [])
            self.attn_mask_buffer = state_dict.get("attn_mask_buffer", [])

    def __iter__(self):
        # Efficiently skip examples if we are resuming
        ds_to_iter = self.dataset
        if self.num_examples_seen > 0:
            if hasattr(ds_to_iter, 'skip'):
                logger.info(f"Utilizing datasets `skip` method to fast forward {self.num_examples_seen} examples.")
                ds_to_iter = ds_to_iter.skip(self.num_examples_seen)
            else:
                # Fallback for datasets without a native skip method
                logger.info(f"Falling back to unoptimized fast forward method for {self.num_examples_seen} examples.")
                it = iter(ds_to_iter)
                for _ in range(self.num_examples_seen):
                    next(it)
                ds_to_iter = it

        for example in ds_to_iter:
            self.num_examples_seen += 1
            tokenized_example = self.tokenizer(example["text"] + self.tokenizer.eos_token, padding=False, truncation=True)

            self.token_id_buffer.extend(tokenized_example["input_ids"])
            self.attn_mask_buffer.extend(tokenized_example["attention_mask"])
            
            # Tokenize in chunks of chunk_size
            while len(self.token_id_buffer) >= self.chunk_size:
                # Extract a chunk
                chunk_ids = self.token_id_buffer[:self.chunk_size]
                chunk_attention = self.attn_mask_buffer[:self.chunk_size]
                
                # Remove used portion from the buffer
                self.token_id_buffer = self.token_id_buffer[self.chunk_size - self.overlap_size:]
                self.attn_mask_buffer = self.attn_mask_buffer[self.chunk_size - self.overlap_size:]
                
                # Convert to tensors and yield
                yield torch.tensor(chunk_ids, dtype=torch.long), torch.tensor(chunk_attention, dtype=torch.long)
        
        # Reset state for the next epoch
        self.num_examples_seen = 0
        self.token_id_buffer = []
        self.attn_mask_buffer = []

class FinetuneRecipeDataset(IterableDataset):
    def __init__(self, data: list, tokenizer: AutoTokenizer, max_length: int, prompt_template: str):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.num_examples_seen = 0

    def state_dict(self):
        """Returns the current state of the dataset for checkpointing."""
        return {"num_examples_seen": self.num_examples_seen}

    def load_state_dict(self, state_dict):
        """Restores the dataset state from a checkpoint."""
        if state_dict:
            self.num_examples_seen = state_dict.get("num_examples_seen", 0)

    def __iter__(self):
        # Slice the list to resume from the last saved index
        if self.num_examples_seen > 0:
            logger.info(f"Fast forwarding to example {self.num_examples_seen + 1}")
        data_to_iter = self.data[self.num_examples_seen:]
        
        for item in data_to_iter:
            self.num_examples_seen += 1
            # Tokenize prompt and full text separately to find the split point
            # Note: item['text'] already includes the prompt from data_factory
            full_text = item['text'] + self.tokenizer.eos_token
            
            # Extract just the prompt portion to calculate its length
            # We assume the prompt ends at the first double newline
            prompt_part = full_text.split("\n\n")[0] + "\n\n"
            
            enc_full = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False)
            enc_prompt = self.tokenizer(prompt_part, truncation=True, max_length=self.max_length, padding=False)
            
            input_ids = torch.tensor(enc_full["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(enc_full["attention_mask"], dtype=torch.long)
            
            # Create Labels: Copy input_ids, then mask the prompt part
            labels = input_ids.clone()
            prompt_len = len(enc_prompt["input_ids"])
            labels[:prompt_len] = -100 # Mask prompt
            
            # Mask padding tokens in labels as well
            labels[attention_mask == 0] = -100
            
            # We return a dict so the collator can handle it easily
            yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # Reset state for the next epoch
        self.num_examples_seen = 0

    def __len__(self):
        return len(self.data)