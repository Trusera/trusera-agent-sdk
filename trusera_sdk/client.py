"""Main client for interacting with the Trusera API."""

import atexit
import logging
import os
import threading
import time
from queue import Empty, Queue
from typing import Any, Optional

import httpx

from .events import Event

try:
    from importlib.metadata import version as _get_version

    _SDK_VERSION = _get_version("trusera-sdk")
except Exception:
    _SDK_VERSION = "0.1.1"

logger = logging.getLogger(__name__)

# Default max retries for failed flush attempts before events are dropped
_MAX_FLUSH_RETRIES = 3


class TruseraClient:
    """
    Client for sending AI agent events to Trusera.

    The client maintains an in-memory queue and flushes events in batches
    to the Trusera API on a background thread.

    Supports env var fallback:
        - ``TRUSERA_API_KEY`` for the API key
        - ``TRUSERA_API_URL`` for the base URL

    Example:
        >>> client = TruseraClient(api_key="tsk_...")
        >>> agent_id = client.register_agent(name="my-agent", framework="langchain")
        >>> client.set_agent_id(agent_id)
        >>> client.track(Event(type=EventType.TOOL_CALL, name="search"))
        >>> client.close()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        flush_interval: float = 5.0,
        batch_size: int = 100,
        timeout: float = 10.0,
        max_retries: int = _MAX_FLUSH_RETRIES,
    ) -> None:
        """
        Initialize the Trusera client.

        Args:
            api_key: Trusera API key (starts with 'tsk_'). Falls back to
                ``TRUSERA_API_KEY`` env var if not provided.
            base_url: Base URL for the Trusera API. Falls back to
                ``TRUSERA_API_URL`` env var, then ``https://api.trusera.dev``.
            flush_interval: Seconds between automatic flushes.
            batch_size: Maximum events per batch.
            timeout: HTTP request timeout in seconds.
            max_retries: Max retry attempts for failed flushes before dropping events.
        """
        resolved_key = api_key or os.environ.get("TRUSERA_API_KEY", "")
        resolved_url = base_url or os.environ.get("TRUSERA_API_URL", "https://api.trusera.dev")

        if not resolved_key:
            raise ValueError(
                "API key is required. Pass api_key= or set TRUSERA_API_KEY env var."
            )

        if not resolved_key.startswith("tsk_"):
            logger.warning("API key should start with 'tsk_' prefix")

        self.api_key = resolved_key
        self.base_url = resolved_url.rstrip("/")
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

        self._queue: Queue[Event] = Queue()
        self._agent_id: Optional[str] = None
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._retry_counts: dict[str, int] = {}

        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"trusera-sdk-python/{_SDK_VERSION}",
            },
            timeout=self.timeout,
        )

        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

        # Register cleanup on exit
        atexit.register(self.close)

    def set_agent_id(self, agent_id: str) -> None:
        """Set the agent ID for this client."""
        with self._lock:
            self._agent_id = agent_id
            logger.info(f"Agent ID set to: {agent_id}")

    def register_agent(
        self, name: str, framework: str, metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Register a new agent with Trusera.

        Args:
            name: Agent name
            framework: Framework name (e.g., "langchain", "crewai", "autogen")
            metadata: Additional agent metadata

        Returns:
            The created agent ID

        Raises:
            httpx.HTTPError: If the API request fails
        """
        payload = {
            "name": name,
            "framework": framework,
            "metadata": metadata or {},
        }

        try:
            response = self._client.post(f"{self.base_url}/api/v1/agents", json=payload)
            response.raise_for_status()
            data = response.json()
            agent_id = data["id"]
            self.set_agent_id(agent_id)
            logger.info(f"Registered agent '{name}' with ID: {agent_id}")
            return agent_id
        except httpx.HTTPError as e:
            logger.error(f"Failed to register agent: {e}")
            raise

    def track(self, event: Event) -> None:
        """
        Add an event to the queue for sending to Trusera.

        Args:
            event: The event to track
        """
        if self._shutdown.is_set():
            logger.warning("Client is shutting down, event will not be tracked")
            return

        self._queue.put(event)
        logger.debug(f"Queued event: {event.type.value} - {event.name}")

        # Flush immediately if we've hit the batch size
        if self._queue.qsize() >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """
        Immediately flush all queued events to the Trusera API.

        This is called automatically on a background thread, but can be
        called manually if you need to ensure events are sent immediately.

        Thread-safe: concurrent flushes are serialized via a lock.
        """
        with self._flush_lock:
            self._flush_once()

    def _flush_once(self) -> None:
        """Internal flush (must be called with _flush_lock held)."""
        if not self._agent_id:
            logger.warning("No agent ID set, cannot flush events")
            return

        events_to_send: list[Event] = []

        # Drain the queue up to batch_size
        while len(events_to_send) < self.batch_size:
            try:
                event = self._queue.get_nowait()
                events_to_send.append(event)
            except Empty:
                break

        if not events_to_send:
            return

        # Send batch to API
        payload = {
            "events": [event.to_dict() for event in events_to_send],
        }

        try:
            url = f"{self.base_url}/api/v1/agents/{self._agent_id}/events"
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            logger.info(f"Flushed {len(events_to_send)} events to Trusera")
            # Reset retry counts on success
            self._retry_counts.clear()
        except httpx.HTTPError as e:
            logger.error(f"Failed to flush events: {e}")
            # Re-queue events only if under retry limit
            batch_key = events_to_send[0].id
            retries = self._retry_counts.get(batch_key, 0) + 1
            if retries < self.max_retries:
                self._retry_counts[batch_key] = retries
                for event in events_to_send:
                    self._queue.put(event)
                logger.warning(
                    f"Re-queued {len(events_to_send)} events (attempt {retries}/{self.max_retries})"
                )
            else:
                logger.error(
                    f"Dropping {len(events_to_send)} events after {self.max_retries} failed attempts"
                )
                self._retry_counts.pop(batch_key, None)

    def _flush_loop(self) -> None:
        """Background thread that periodically flushes events."""
        while not self._shutdown.is_set():
            time.sleep(self.flush_interval)
            if not self._shutdown.is_set():
                self.flush()

    def close(self) -> None:
        """
        Close the client and flush any remaining events.

        This is called automatically on exit via atexit.
        """
        if self._shutdown.is_set():
            return

        logger.info("Closing Trusera client...")
        self._shutdown.set()

        # Wait for flush thread to exit
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=self.flush_interval + 1)

        # Final flush
        self.flush()

        # Close HTTP client
        self._client.close()
        logger.info("Trusera client closed")

    def __enter__(self) -> "TruseraClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()


class AsyncTruseraClient:
    """
    Async client for sending AI agent events to Trusera.

    Provides an asyncio-native interface. Events are queued locally and
    flushed to the API using ``httpx.AsyncClient``.

    Example:
        >>> async with AsyncTruseraClient(api_key="tsk_...") as client:
        ...     await client.register_agent("my-agent", "langchain")
        ...     client.track(Event(type=EventType.TOOL_CALL, name="search"))
        ...     await client.flush()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 100,
        timeout: float = 10.0,
        max_retries: int = _MAX_FLUSH_RETRIES,
    ) -> None:
        """
        Initialize the async Trusera client.

        Args:
            api_key: Trusera API key (starts with 'tsk_'). Falls back to
                ``TRUSERA_API_KEY`` env var.
            base_url: Base URL for the Trusera API. Falls back to
                ``TRUSERA_API_URL`` env var, then ``https://api.trusera.dev``.
            batch_size: Maximum events per batch.
            timeout: HTTP request timeout in seconds.
            max_retries: Max retry attempts for failed flushes before dropping events.
        """
        resolved_key = api_key or os.environ.get("TRUSERA_API_KEY", "")
        resolved_url = base_url or os.environ.get("TRUSERA_API_URL", "https://api.trusera.dev")

        if not resolved_key:
            raise ValueError(
                "API key is required. Pass api_key= or set TRUSERA_API_KEY env var."
            )

        if not resolved_key.startswith("tsk_"):
            logger.warning("API key should start with 'tsk_' prefix")

        self.api_key = resolved_key
        self.base_url = resolved_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

        self._events: list[Event] = []
        self._agent_id: Optional[str] = None
        self._closed = False
        self._retry_count = 0

        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"trusera-sdk-python/{_SDK_VERSION}",
            },
            timeout=self.timeout,
        )

    def set_agent_id(self, agent_id: str) -> None:
        """Set the agent ID for this client."""
        self._agent_id = agent_id
        logger.info(f"Agent ID set to: {agent_id}")

    async def register_agent(
        self, name: str, framework: str, metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Register a new agent with Trusera.

        Args:
            name: Agent name
            framework: Framework name
            metadata: Additional agent metadata

        Returns:
            The created agent ID

        Raises:
            httpx.HTTPError: If the API request fails
        """
        payload = {
            "name": name,
            "framework": framework,
            "metadata": metadata or {},
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/api/v1/agents", json=payload
            )
            response.raise_for_status()
            data = response.json()
            agent_id = data["id"]
            self.set_agent_id(agent_id)
            logger.info(f"Registered agent '{name}' with ID: {agent_id}")
            return agent_id
        except httpx.HTTPError as e:
            logger.error(f"Failed to register agent: {e}")
            raise

    def track(self, event: Event) -> None:
        """
        Add an event to the local buffer.

        Args:
            event: The event to track
        """
        if self._closed:
            logger.warning("Client is closed, event will not be tracked")
            return

        self._events.append(event)
        logger.debug(f"Queued event: {event.type.value} - {event.name}")

    async def flush(self) -> None:
        """Flush all buffered events to the Trusera API."""
        if not self._agent_id:
            logger.warning("No agent ID set, cannot flush events")
            return

        if not self._events:
            return

        # Take up to batch_size events
        batch = self._events[: self.batch_size]
        self._events = self._events[self.batch_size :]

        payload = {
            "events": [event.to_dict() for event in batch],
        }

        try:
            url = f"{self.base_url}/api/v1/agents/{self._agent_id}/events"
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            logger.info(f"Flushed {len(batch)} events to Trusera")
            self._retry_count = 0
        except httpx.HTTPError as e:
            logger.error(f"Failed to flush events: {e}")
            self._retry_count += 1
            if self._retry_count < self.max_retries:
                self._events = batch + self._events
                logger.warning(
                    f"Re-queued {len(batch)} events "
                    f"(attempt {self._retry_count}/{self.max_retries})"
                )
            else:
                logger.error(
                    f"Dropping {len(batch)} events after {self.max_retries} failed attempts"
                )
                self._retry_count = 0

    async def close(self) -> None:
        """Close the client and flush remaining events."""
        if self._closed:
            return

        logger.info("Closing async Trusera client...")
        self._closed = True
        await self.flush()
        await self._client.aclose()
        logger.info("Async Trusera client closed")

    async def __aenter__(self) -> "AsyncTruseraClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
