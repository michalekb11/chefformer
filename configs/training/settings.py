from pydantic import model_validator
from pydantic_settings import BaseSettings
from configs.shared.settings import BasePromptSettings, ModelSettings, get_latest_checkpoint

class BaseTrainingArgs(BaseSettings):
    batch_size: int = 6
    checkpoint_dir: str = "./checkpoints"
    gradient_clipping: float = 1.0
    validation_loop_steps: int | None = None

class PreTrainingArgs(BaseTrainingArgs):
   """
   Goal: ~150 Million tokens total. 
    Fast execution, green memory footprint, and visible learning.
    
    Mathematical breakdown:
    - Tokens per iter: 6 batch_size * 50 accumulation * 512 seq_len = 153,600 tokens.
    - Total iters: 3,000 optimizer steps.
    - Total global steps: 3000 * 50 = 150,000 global steps.
    - Total tokens: 3,000 * 153,600 = 460,800,000 tokens (~461M).
   """
   learning_rate: float = 0.0004 # max learning rate if using scheduler with warm up and decay, originally 0.00005
   num_epochs: int = 1
   #weight_decay: float = 0.0005 # using default weight decay and betas
   gradient_accumulation_steps: int = 50
   warmup_iters: int = 300 # gradient_accumulation_steps per iter
   save_checkpoint_every: int = 7500 # (100 iters)
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
    
class FinetuningArgs(BaseTrainingArgs):
   """
   Training arguments for recipe generation for finetuning
   """
   learning_rate: float = 0.00004
   num_epochs: int = 2
   gradient_accumulation_steps: int = 16
   warmup_iters: int = 180 # gradient_accumulation_steps per iter
   save_checkpoint_every: int = 5000 # (~312 iters)
   decay_start_iter: int = 180
   decay_total_iters: int = 1702
   validate_every: int = 2500 # (~156 iters)

class FinetuningSettings(BaseSettings):
    task: str = "finetune"
    checkpoint_path: str | None = None
    model: ModelSettings = ModelSettings()
    training_args: FinetuningArgs = FinetuningArgs()
    prompt: BasePromptSettings = BasePromptSettings()

    @model_validator(mode='after')
    def set_default_checkpoint(self) -> 'FinetuningSettings':
        """Automatically discovers the latest checkpoint if path is not explicitly provided."""
        if self.checkpoint_path is None:
            self.checkpoint_path = get_latest_checkpoint(self.training_args.checkpoint_dir, self.task)
        return self

pretraining_settings = PreTrainingSettings()
finetuning_settings = FinetuningSettings()

if __name__ == '__main__':
    print(finetuning_settings)