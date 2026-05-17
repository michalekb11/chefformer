from pydantic import model_validator
from pydantic_settings import BaseSettings
from configs.shared.settings import ModelSettings, get_latest_checkpoint

class PreTrainingArgs(BaseSettings):
   """
   Goal: ~150 Million tokens total. 
    Fast execution, green memory footprint, and visible learning.
    
    Mathematical breakdown:
    - Tokens per iter: 6 batch_size * 50 accumulation * 512 seq_len = 153,600 tokens.
    - Total iters: 1,000 optimizer steps.
    - Total global steps: 1000 * 50 = 50,000 global steps.
    - Total tokens: 1,000 * 131,072 = 131,072,000 tokens (~131M).
   """
   batch_size: int = 6
   learning_rate: float = 0.0004 # max learning rate if using scheduler with warm up and decay, originally 0.00005
   num_epochs: int = 1
   #weight_decay: float = 0.0005 # using default weight decay and betas
   gradient_clipping: float = 1.0
   gradient_accumulation_steps: int = 50
   warmup_iters: int = 300 # gradient_accumulation_steps per iter
   save_checkpoint_every: int = 7500 # (100 iters)
   checkpoint_dir: str = "./checkpoints"
   decay_start_iter: int = 300
   decay_total_iters: int = 2700
   validation_loop_steps: int = 800 # (equivalent of 8 iters)
   validate_every: int = 7500 # (3000 iters)

class PreTrainingSettings(BaseSettings):
    task: str = "pretrain"
    checkpoint_path: str | None = None
    model: ModelSettings = ModelSettings()
    training_args: PreTrainingArgs = PreTrainingArgs()

    @model_validator(mode='after')
    def set_default_checkpoint(self) -> 'PreTrainingSettings':
        """Automatically discovers the latest checkpoint if path is not explicitly provided."""
        if self.checkpoint_path is None:
            self.checkpoint_path = get_latest_checkpoint(self.training_args.checkpoint_dir, self.task)
        return self


pretraining_settings = PreTrainingSettings()

if __name__ == '__main__':
    print(get_latest_checkpoint('./checkpoints', 'pretrain'))