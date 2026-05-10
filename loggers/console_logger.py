import logging
import sys
from typing import Any, Dict
from loggers.base import MetricLogger, AppLogger

class ConsoleLogger(MetricLogger, AppLogger):
    def __init__(self, name: str = "Chefformer"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Standard format: [2023-10-27 10:00:00] [INFO] Message
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_metrics(self, step: int, metrics: Dict[str, Any], phase: str = "train"):
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"[{phase.upper()}] Step {step} | {metrics_str}")

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)
