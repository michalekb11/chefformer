import logging
import sys
import os
from typing import Any, Dict
from loggers.base import MetricLogger, AppLogger

class ConsoleLogger(MetricLogger, AppLogger):
    def __init__(self, name: str = "Chefformer", log_file: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers if called multiple times (e.g., in notebooks or during restarts)
        if not self.logger.handlers:
            # Standard format: [2023-10-27 10:00:00] [INFO] Message
            formatter = logging.Formatter(
                fmt='[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Console Handler
            console_h = logging.StreamHandler(sys.stdout)
            console_h.setFormatter(formatter)
            self.logger.addHandler(console_h)
            
            # File Handler (if log_file is provided)
            if log_file:
                log_dir = os.path.dirname(log_file)
                if log_dir: # Ensure directory exists if path is not just a filename
                    os.makedirs(log_dir, exist_ok=True)
                file_h = logging.FileHandler(log_file)
                file_h.setFormatter(formatter)
                self.logger.addHandler(file_h)

    def log_metrics(self, step: int, metrics: Dict[str, Any], task: str, prefix: str = ""):
        metrics_str = " | ".join([f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        tag = f"{task.upper()}:{prefix.upper()}" if prefix else task.upper()
        self.logger.info(f"[{tag}] Step {step} | {metrics_str}")

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)
