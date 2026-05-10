from pydantic_settings import BaseSettings
from configs.shared.settings import ModelSettings

class PreTrainingArgs(BaseSettings):
   batch_size: int = 8
   learning_rate: float = 0.00007 # max learning rate if using scheduler with warm up and decay, originally 0.00005
   num_epochs: int = 1
   weight_decay: float = 0.0005
   gradient_clipping: float = 5.0
   gradient_accumulation_steps: int = 10 # originally 5
   warmup_iters: int = 800 # gradient_accumulation_steps per iter
   save_checkpoint_every: int = 1000
   checkpoint_dir: str = "./checkpoints"
   decay_start_iter: int = 4000
   decay_total_iters: int = 3000
   validation_loop_steps: int = 800
   validate_every: int = 2000

class PreTrainingSettings(BaseSettings):
    task: str = "pretrain"
    model: ModelSettings = ModelSettings()
    training_args: PreTrainingArgs = PreTrainingArgs()

pretraining_settings = PreTrainingSettings()

if __name__ == '__main__':
    print(pretraining_settings)