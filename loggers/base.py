from abc import ABC, abstractmethod
from typing import Any, Dict

class MetricLogger(ABC):
    """Interface for telemetry/metrics (Loss, Accuracy, etc.)."""
    @abstractmethod
    def log_metrics(self, step: int, metrics: Dict[str, Any], task: str, prefix: str= ""):
        pass

class AppLogger(ABC):
    """Interface for application events (Info, Warning, Error)."""
    @abstractmethod
    def info(self, message: str):
        pass

    @abstractmethod
    def warning(self, message: str):
        pass

    @abstractmethod
    def error(self, message: str):
        pass
