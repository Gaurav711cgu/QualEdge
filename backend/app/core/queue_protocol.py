"""
Abstract base class definition for async queue providers.
Allows swapping SQLite transactional outbox with Celery/Redis with zero service-layer code changes.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseQueueProvider(ABC):
    @abstractmethod
    def enqueue(
        self, 
        event_id: str, 
        event_type: str, 
        payload: Dict[str, Any], 
        connection: Optional[Any] = None
    ) -> None:
        """
        Enqueues an action to be processed asynchronously.
        If a connection parameter is supplied (e.g. SQLite connection), 
        the event must be queued inside the active database transaction.
        """
        pass

    @abstractmethod
    def start_worker(self, callback: Any) -> None:
        """
        Starts the background worker to monitor the queue and process tasks.
        """
        pass
