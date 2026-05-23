import random
import gc
from tqdm import tqdm
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from src.training.utils.data_utils import ChunkedTextDataset, FinetuneRecipeDataset
from src.training.utils.training_loop_utils import batch_collator
from src.training.database.MongoDBClient import MongoDBClient
from src.training.database.Recipe import Recipe
from loggers.console_logger import ConsoleLogger

logger = ConsoleLogger(__name__)

def get_dataloaders(task, tokenizer, settings, base_prompt: str=None):
    """
    Factory to return training and validation dataloaders.
    task: 'pretrain' (C4) or 'finetune' (Mongo Recipes)
    """
    if task == 'pretrain':
        train_ds = load_dataset("allenai/c4", "en", streaming=True, split='train')
        val_ds = load_dataset("allenai/c4", "en", streaming=True, split='validation')

        # Wrap in Chunked Logic for Pretraining
        train_loader_ds = ChunkedTextDataset(
            train_ds, tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=500
        )
        val_loader_ds = ChunkedTextDataset(
            val_ds, tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=500
        )
    elif task == 'finetune':
        if not base_prompt:
            raise ValueError("Recipe generation finetuning requires a base prompt.")
        
        mongo = MongoDBClient()
        raw_recipes = [Recipe.from_dict(doc) for doc in mongo.collection.find()]
        
        processed_data = []
        logger.info(f"Filtering {len(raw_recipes)} recipes for length...")
       
        for r in tqdm(raw_recipes):
            text = f"{base_prompt} {r.title}\n\n{r.to_string()}"
            tokens = tokenizer.encode(text)
            length = len(tokens)

            # Filter to 511 to fit within 512 context limit (EOS token added by dataset)
            if length <= 511:
                processed_data.append({"text": text, "length": length})
        
        # Clear raw objects immediately to free Host RAM
        del raw_recipes
        gc.collect()
        
        # Shuffle and split into 90% training and 10% validation
        random.seed(42)
        random.shuffle(processed_data)
        
        split_idx = int(len(processed_data) * 0.9)
        train_ds = processed_data[:split_idx]
        val_ds = processed_data[split_idx:]

        # Optimization: Grouped Batching to minimize padding.
        # Sort training data by length.
        train_ds.sort(key=lambda x: x['length'])
        
        # Form batches and shuffle the order of batches.
        batch_size = settings.batch_size
        train_batches = [train_ds[i : i + batch_size] for i in range(0, len(train_ds), batch_size)]
        
        random.seed(42)
        random.shuffle(train_batches)
        train_ds = [item for batch in train_batches for item in batch]

        # Use dedicated Recipe Dataset (No chunking/sliding window)
        train_loader_ds = FinetuneRecipeDataset(train_ds, tokenizer, max_length=512, prompt_template=base_prompt)
        val_loader_ds = FinetuneRecipeDataset(val_ds, tokenizer, max_length=512, prompt_template=base_prompt)

    train_loader = StatefulDataLoader(
        train_loader_ds, 
        batch_size=settings.batch_size, 
        collate_fn=batch_collator, 
        num_workers=0,
        snapshot_every_n_steps=settings.save_checkpoint_every
    )
    
    val_loader = StatefulDataLoader(
        val_loader_ds, 
        batch_size=settings.batch_size, 
        collate_fn=batch_collator, 
        num_workers=0
    )

    return train_loader, val_loader
