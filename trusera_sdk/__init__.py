"""Trusera SDK for monitoring AI agents."""

from .client import AsyncTruseraClient, TruseraClient
from .decorators import get_default_client, monitor, set_default_client
from .events import Event, EventType

__version__ = "0.1.1"

__all__ = [
    "TruseraClient",
    "AsyncTruseraClient",
    "Event",
    "EventType",
    "monitor",
    "set_default_client",
    "get_default_client",
]
