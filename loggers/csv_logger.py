import os
import pandas as pd
from typing import Any, Dict
from loggers.base import MetricLogger

class CSVLogger(MetricLogger):
    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def prepare_for_resume(self, start_step: int):
        """
        Truncates the single CSV file to remove any rows after the start_step.
        """
        if os.path.exists(self.filepath) and start_step > 0:
            try:
                df = pd.read_csv(self.filepath)
                if 'step' in df.columns:
                    df = df[df['step'] <= start_step]
                    df.to_csv(self.filepath, index=False)
            except Exception as e:
                print(f"Warning: Could not truncate {self.filepath}: {e}")

    def log_metrics(self, step: int, metrics: Dict[str, Any], task: str, prefix: str):
        """
        Logs metrics to a single CSV file, using 'prefix' as the phase.
        """
        new_row = {
            'step': step,
            'phase': prefix,
            **metrics
        }
        
        df_new = pd.DataFrame([new_row])
        
        if not os.path.exists(self.filepath):
            df_new.to_csv(self.filepath, header=True, index=False)
        else:
            df_new.to_csv(self.filepath, mode='a', header=False, index=False)
