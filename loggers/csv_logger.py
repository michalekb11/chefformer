import os
import csv
from typing import Any, Dict
from loggers.base import MetricLogger

class CSVLogger(MetricLogger):
    def __init__(self, log_dir: str = "./checkpoints"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_metrics(self, step: int, metrics: Dict[str, Any], task: str, prefix: str):
        """
        Logs metrics to CSV files. 
        Each key in the metrics dict gets its own column/file or grouped by phase.
        """
        for key, value in metrics.items():
            os.makedirs(f"{self.log_dir}/{task}", exist_ok=True)
            filename = os.path.join(self.log_dir, f"{task}/{prefix}_{key}.csv")
            
            # Initialize file with header if it doesn't exist
            file_exists = os.path.isfile(filename)
            
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["step", key])
                writer.writerow([step, value])
