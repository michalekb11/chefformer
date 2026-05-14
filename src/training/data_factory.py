from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from src.training.utils.data_utils import ChunkedTextDataset
from src.training.utils.training_loop_utils import batch_collator

def get_dataloaders(task, tokenizer, settings):
    """
    Factory to return training and validation dataloaders.
    task: 'pretrain' (C4) or 'finetune' (Mongo Recipes)
    """
    if task == 'pretrain':
        train_ds = load_dataset("allenai/c4", "en", streaming=True, split='train')
        val_ds = load_dataset("allenai/c4", "en", streaming=True, split='validation')
    elif task == 'finetune':
        # Future logic for Mongo loading
        # train_ds = MongoDataset(...)
        raise NotImplementedError("Mongo finetuning dataset logic goes here.")
    
    # Wrap in Chunked Logic
    train_chunked = ChunkedTextDataset(
        train_ds, tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=500
    )
    val_chunked = ChunkedTextDataset(
        val_ds, tokenizer, chunk_size=512, overlap_size=50, shuffle=True, shuffle_buffer_size=500
    )

    train_loader = StatefulDataLoader(
        train_chunked, 
        batch_size=settings.batch_size, 
        collate_fn=batch_collator, 
        num_workers=0,
        snapshot_every_n_steps=settings.save_checkpoint_every
    )
    
    val_loader = StatefulDataLoader(
        val_chunked, 
        batch_size=settings.batch_size, 
        collate_fn=batch_collator, 
        num_workers=0
    )

    return train_loader, val_loader
