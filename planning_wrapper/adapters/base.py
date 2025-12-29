from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTaskAdapter(ABC):
    @abstractmethod
    def get_task_state(self, env: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def set_task_state(self, env: Any, task_state: Dict[str, Any]) -> None:
        raise NotImplementedError