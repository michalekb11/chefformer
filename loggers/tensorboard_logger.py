from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict
from loggers.base import MetricLogger

class TensorBoardLogger(MetricLogger):
    def __init__(self, log_dir: str, purge_step: int = None):
        """
        Initializes the TensorBoard SummaryWriter.
        purge_step ensures that if we restart from a checkpoint, 
        any logs after that step are removed from the visualization.
        """
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step)

    def log_metrics(self, step: int, metrics: Dict[str, Any], task: str, prefix: str = ""):
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            if isinstance(v, (int, float)):
                self.writer.add_scalar(tag, v, step)

    def close(self):
        self.writer.close()