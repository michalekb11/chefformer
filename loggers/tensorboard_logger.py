from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict
from loggers.base import MetricLogger

class TensorBoardLogger(MetricLogger):
    def __init__(self, log_dir: str, purge_step: int = None):
        # purge_step tells TensorBoard to discard any logs after this step (useful for restarts)
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step)

    def log_metrics(self, step: int, metrics: Dict[str, Any], task: str = "", prefix: str = ""):
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, v, step)

    def close(self):
        self.writer.close()