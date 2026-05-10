from typing import Any, Dict, List
from loggers.base import MetricLogger

class CompositeMetricLogger(MetricLogger):
    """Allows logging metrics to multiple destinations simultaneously."""
    def __init__(self, loggers: List[MetricLogger]):
        self.loggers = loggers

    def log_metrics(self, step: int, metrics: Dict[str, Any], task: str, prefix: str = ""):
        for logger in self.loggers:
            logger.log_metrics(step, metrics, task, prefix)
