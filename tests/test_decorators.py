"""Tests for decorator functionality."""

import asyncio
from datetime import datetime, timezone
from enum import Enum

import pytest

from trusera_sdk import EventType, monitor, set_default_client
from trusera_sdk.decorators import _serialize_value, _MAX_PAYLOAD_SIZE


def test_monitor_sync_function(trusera_client):
    """Test @monitor decorator on sync function."""
    set_default_client(trusera_client)

    @monitor()
    def test_function(x: int, y: int) -> int:
        return x + y

    result = test_function(2, 3)

    assert result == 5
    assert trusera_client._queue.qsize() == 1

    # Check event
    event = trusera_client._queue.get()
    assert event.type == EventType.TOOL_CALL
    assert event.name == "test_function"
    assert event.payload["arguments"]["x"] == 2
    assert event.payload["arguments"]["y"] == 3
    assert event.payload["result"] == 5
    assert event.metadata["success"] is True
    assert event.metadata["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_monitor_async_function(trusera_client):
    """Test @monitor decorator on async function."""
    set_default_client(trusera_client)

    @monitor()
    async def async_function(value: str) -> str:
        await asyncio.sleep(0.01)
        return value.upper()

    result = await async_function("hello")

    assert result == "HELLO"
    assert trusera_client._queue.qsize() == 1

    # Check event
    event = trusera_client._queue.get()
    assert event.type == EventType.TOOL_CALL
    assert event.name == "async_function"
    assert event.payload["arguments"]["value"] == "hello"
    assert event.payload["result"] == "HELLO"


def test_monitor_with_custom_name(trusera_client):
    """Test @monitor with custom event name."""
    set_default_client(trusera_client)

    @monitor(name="custom_operation")
    def test_function() -> str:
        return "done"

    test_function()

    event = trusera_client._queue.get()
    assert event.name == "custom_operation"


def test_monitor_with_event_type(trusera_client):
    """Test @monitor with custom event type."""
    set_default_client(trusera_client)

    @monitor(event_type=EventType.DATA_ACCESS)
    def query_database() -> list[str]:
        return ["result1", "result2"]

    query_database()

    event = trusera_client._queue.get()
    assert event.type == EventType.DATA_ACCESS


def test_monitor_without_capture_args(trusera_client):
    """Test @monitor with capture_args=False."""
    set_default_client(trusera_client)

    @monitor(capture_args=False)
    def test_function(secret: str) -> str:
        return secret.upper()

    test_function("password123")

    event = trusera_client._queue.get()
    assert "arguments" not in event.payload


def test_monitor_without_capture_result(trusera_client):
    """Test @monitor with capture_result=False."""
    set_default_client(trusera_client)

    @monitor(capture_result=False)
    def test_function() -> str:
        return "sensitive_data"

    test_function()

    event = trusera_client._queue.get()
    assert "result" not in event.payload


def test_monitor_with_exception(trusera_client):
    """Test @monitor capturing exceptions."""
    set_default_client(trusera_client)

    @monitor()
    def failing_function() -> None:
        raise ValueError("Something went wrong")

    with pytest.raises(ValueError):
        failing_function()

    event = trusera_client._queue.get()
    assert event.metadata["success"] is False
    assert event.payload["error"]["type"] == "ValueError"
    assert event.payload["error"]["message"] == "Something went wrong"


@pytest.mark.asyncio
async def test_monitor_async_with_exception(trusera_client):
    """Test @monitor on async function with exception."""
    set_default_client(trusera_client)

    @monitor()
    async def async_failing() -> None:
        await asyncio.sleep(0.01)
        raise RuntimeError("Async error")

    with pytest.raises(RuntimeError):
        await async_failing()

    event = trusera_client._queue.get()
    assert event.metadata["success"] is False
    assert event.payload["error"]["type"] == "RuntimeError"


def test_monitor_with_explicit_client(trusera_client):
    """Test @monitor with explicitly passed client."""
    # Don't set default client

    @monitor(client=trusera_client)
    def test_function() -> str:
        return "test"

    test_function()

    assert trusera_client._queue.qsize() == 1


def test_monitor_without_client(caplog):
    """Test @monitor without any client configured."""
    set_default_client(None)

    @monitor()
    def test_function() -> str:
        return "test"

    result = test_function()

    assert result == "test"
    assert "No Trusera client configured" in caplog.text


def test_monitor_preserves_function_metadata():
    """Test that @monitor preserves function metadata."""
    @monitor()
    def documented_function(x: int) -> int:
        """This is a docstring."""
        return x * 2

    assert documented_function.__name__ == "documented_function"
    assert documented_function.__doc__ == "This is a docstring."


def test_monitor_with_complex_types(trusera_client):
    """Test @monitor with complex argument types."""
    set_default_client(trusera_client)

    @monitor()
    def process_data(data: dict[str, list[int]]) -> int:
        return sum(data.get("numbers", []))

    result = process_data({"numbers": [1, 2, 3]})

    assert result == 6
    event = trusera_client._queue.get()
    assert event.payload["arguments"]["data"]["numbers"] == [1, 2, 3]


# --- New tests for extended serialization ---


def test_serialize_bytes():
    """Test _serialize_value handles bytes."""
    assert _serialize_value(b"hello") == "<bytes len=5>"
    assert _serialize_value(b"") == "<bytes len=0>"


def test_serialize_set():
    """Test _serialize_value handles sets (sorted list output)."""
    result = _serialize_value({3, 1, 2})
    assert result == [1, 2, 3]


def test_serialize_datetime():
    """Test _serialize_value handles datetime and date."""
    dt = datetime(2026, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
    assert _serialize_value(dt) == "2026-02-14T12:00:00+00:00"

    from datetime import date
    d = date(2026, 2, 14)
    assert _serialize_value(d) == "2026-02-14"


def test_serialize_enum():
    """Test _serialize_value handles Enum types."""
    class Color(Enum):
        RED = "red"
        BLUE = "blue"

    assert _serialize_value(Color.RED) == "red"
    assert _serialize_value(EventType.TOOL_CALL) == "tool_call"


def test_monitor_with_extended_types(trusera_client):
    """Test @monitor captures bytes, set, datetime, Enum in args."""
    set_default_client(trusera_client)

    class Status(Enum):
        ACTIVE = "active"

    @monitor()
    def process(data: bytes, tags: set, status: Status) -> str:
        return "ok"

    process(b"binary", {"a", "b"}, Status.ACTIVE)

    event = trusera_client._queue.get()
    assert event.payload["arguments"]["data"] == "<bytes len=6>"
    assert sorted(event.payload["arguments"]["tags"]) == ["a", "b"]
    assert event.payload["arguments"]["status"] == "active"


def test_payload_truncation(trusera_client):
    """Test that oversized payloads are truncated."""
    set_default_client(trusera_client)

    @monitor()
    def big_output() -> str:
        return "x" * (_MAX_PAYLOAD_SIZE + 1000)

    big_output()

    event = trusera_client._queue.get()
    assert event.payload.get("_truncated") is True or len(str(event.payload)) <= _MAX_PAYLOAD_SIZE + 1000
