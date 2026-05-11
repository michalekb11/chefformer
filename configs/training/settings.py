import os
import re
from pydantic import model_validator
from pydantic_settings import BaseSettings
from configs.shared.settings import ModelSettings

def get_latest_checkpoint(checkpoint_dir: str, task: str) -> str | None:
    """Helper to find the most recent checkpoint based on epoch and step."""
    target_dir = os.path.join(checkpoint_dir, task)
    if not os.path.exists(target_dir):
        return None

    # Pattern to capture epoch and step: epoch{E}_step{S}.pth
    pattern = re.compile(r"epoch(\d+)_step(\d+)\.pth")
    
    checkpoints = []
    for filename in os.listdir(target_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            checkpoints.append((epoch, step, filename))

    if not checkpoints:
        return None

    # Sort descending by epoch, then step
    checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return os.path.join(target_dir, checkpoints[0][2])

class PreTrainingArgs(BaseSettings):
   batch_size: int = 16
   learning_rate: float = 0.00007 # max learning rate if using scheduler with warm up and decay, originally 0.00005
   num_epochs: int = 1
   weight_decay: float = 0.0005
   gradient_clipping: float = 5.0
   gradient_accumulation_steps: int = 5
   warmup_iters: int = 800 # gradient_accumulation_steps per iter
   save_checkpoint_every: int = 1000
   checkpoint_dir: str = "./checkpoints"
   decay_start_iter: int = 4000
   decay_total_iters: int = 3000
   validation_loop_steps: int = 800
   validate_every: int = 2000

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