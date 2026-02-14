"""Tests for TruseraClient."""

import os
import time

import pytest
from unittest.mock import Mock

from trusera_sdk import TruseraClient, AsyncTruseraClient, Event, EventType


def test_client_initialization(mock_httpx_client):
    """Test TruseraClient initialization."""
    client = TruseraClient(
        api_key="tsk_test_key",
        base_url="https://api.test.trusera.dev",
    )

    assert client.api_key == "tsk_test_key"
    assert client.base_url == "https://api.test.trusera.dev"
    assert client.flush_interval == 5.0
    assert client.batch_size == 100

    client.close()


def test_client_api_key_warning(mock_httpx_client, caplog):
    """Test warning for API key without tsk_ prefix."""
    client = TruseraClient(api_key="invalid_key")

    assert "should start with 'tsk_'" in caplog.text

    client.close()


def test_client_requires_api_key(mock_httpx_client):
    """Test that client raises ValueError when no API key is provided."""
    with pytest.raises(ValueError, match="API key is required"):
        TruseraClient()


def test_client_env_var_fallback(mock_httpx_client, monkeypatch):
    """Test TRUSERA_API_KEY and TRUSERA_API_URL env var fallback."""
    monkeypatch.setenv("TRUSERA_API_KEY", "tsk_from_env")
    monkeypatch.setenv("TRUSERA_API_URL", "https://custom.trusera.dev")

    client = TruseraClient()

    assert client.api_key == "tsk_from_env"
    assert client.base_url == "https://custom.trusera.dev"

    client.close()


def test_client_explicit_key_overrides_env(mock_httpx_client, monkeypatch):
    """Test that explicit api_key takes precedence over env var."""
    monkeypatch.setenv("TRUSERA_API_KEY", "tsk_from_env")

    client = TruseraClient(api_key="tsk_explicit")

    assert client.api_key == "tsk_explicit"

    client.close()


def test_set_agent_id(trusera_client):
    """Test setting agent ID."""
    trusera_client.set_agent_id("agent_new_456")
    assert trusera_client._agent_id == "agent_new_456"


def test_register_agent(mock_httpx_client):
    """Test agent registration."""
    client = TruseraClient(api_key="tsk_test_key")

    agent_id = client.register_agent(
        name="test-agent",
        framework="langchain",
        metadata={"version": "1.0"},
    )

    assert agent_id == "agent_123"
    assert client._agent_id == "agent_123"

    # Verify API call
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert "/api/v1/agents" in call_args[0][0]
    assert call_args[1]["json"]["name"] == "test-agent"
    assert call_args[1]["json"]["framework"] == "langchain"

    client.close()


def test_track_event(trusera_client):
    """Test tracking an event."""
    event = Event(
        type=EventType.TOOL_CALL,
        name="test_tool",
        payload={"input": "test"},
    )

    trusera_client.track(event)

    assert trusera_client._queue.qsize() == 1


def test_track_multiple_events(trusera_client):
    """Test tracking multiple events."""
    for i in range(3):
        event = Event(
            type=EventType.TOOL_CALL,
            name=f"tool_{i}",
        )
        trusera_client.track(event)

    assert trusera_client._queue.qsize() == 3


def test_flush_events(trusera_client, mock_httpx_client):
    """Test flushing events to API."""
    # Track some events
    for i in range(3):
        event = Event(
            type=EventType.TOOL_CALL,
            name=f"tool_{i}",
        )
        trusera_client.track(event)

    # Flush
    trusera_client.flush()

    # Verify API call
    assert mock_httpx_client.post.call_count >= 1

    # Find the call that posted events
    events_posted = False
    for call_obj in mock_httpx_client.post.call_args_list:
        if "/events" in call_obj[0][0]:
            events_posted = True
            payload = call_obj[1]["json"]
            assert "events" in payload
            assert len(payload["events"]) == 3
            break

    assert events_posted, "Events were not posted to API"
    assert trusera_client._queue.qsize() == 0


def test_flush_respects_batch_size(mock_httpx_client):
    """Test that flush respects batch_size."""
    # Use a longer flush_interval to prevent background thread interference
    client = TruseraClient(
        api_key="tsk_test_key",
        flush_interval=60.0,
        batch_size=5,
    )
    client.set_agent_id("agent_test_123")

    # Track 10 events (2x batch_size) without triggering auto-flush
    for i in range(10):
        event = Event(type=EventType.TOOL_CALL, name=f"tool_{i}")
        client._queue.put(event)  # Put directly to avoid auto-flush trigger

    # Single flush should drain exactly batch_size
    client.flush()

    # Should have sent 5 events, 5 remaining
    assert client._queue.qsize() == 5

    client.close()


def test_auto_flush_on_batch_size(trusera_client, mock_httpx_client):
    """Test automatic flush when batch_size is reached."""
    # Track events up to batch_size (5)
    for i in range(5):
        event = Event(
            type=EventType.TOOL_CALL,
            name=f"tool_{i}",
        )
        trusera_client.track(event)

    # Should trigger automatic flush
    time.sleep(0.1)  # Give it time to flush

    # Queue might not be empty immediately due to threading
    # but flush should have been called
    assert mock_httpx_client.post.call_count >= 1


def test_flush_without_agent_id(mock_httpx_client, caplog):
    """Test flush without setting agent ID."""
    client = TruseraClient(api_key="tsk_test_key")

    event = Event(type=EventType.TOOL_CALL, name="test")
    client.track(event)

    client.flush()

    assert "No agent ID set" in caplog.text

    client.close()


def test_client_context_manager(mock_httpx_client):
    """Test using client as context manager."""
    with TruseraClient(api_key="tsk_test_key") as client:
        client.set_agent_id("agent_test")
        event = Event(type=EventType.TOOL_CALL, name="test")
        client.track(event)

    # Client should be closed after exiting context
    assert client._shutdown.is_set()


def test_client_close(trusera_client, mock_httpx_client):
    """Test client close method."""
    # Track an event
    event = Event(type=EventType.TOOL_CALL, name="test")
    trusera_client.track(event)

    # Close
    trusera_client.close()

    # Should be shut down
    assert trusera_client._shutdown.is_set()

    # HTTP client should be closed
    mock_httpx_client.close.assert_called()


def test_client_close_idempotent(trusera_client):
    """Test that close can be called multiple times safely."""
    trusera_client.close()
    trusera_client.close()  # Should not raise


def test_track_after_shutdown(trusera_client, caplog):
    """Test tracking after client is shut down."""
    trusera_client.close()

    event = Event(type=EventType.TOOL_CALL, name="test")
    trusera_client.track(event)

    assert "shutting down" in caplog.text.lower()


def test_flush_retry_limit(mock_httpx_client):
    """Test that flush drops events after max_retries."""
    import httpx as _httpx

    mock_httpx_client.post.side_effect = _httpx.ConnectError("connection failed")

    client = TruseraClient(
        api_key="tsk_test_key",
        flush_interval=60.0,
        batch_size=100,
        max_retries=2,
    )
    client.set_agent_id("agent_test")

    event = Event(type=EventType.TOOL_CALL, name="test")
    client._queue.put(event)

    # First flush: attempt 1 fails, re-queues
    client.flush()
    assert client._queue.qsize() == 1

    # Second flush: attempt 2 fails, drops events
    client.flush()
    assert client._queue.qsize() == 0

    client.close()


# --- AsyncTruseraClient tests ---


@pytest.mark.asyncio
async def test_async_client_initialization(mock_httpx_client):
    """Test AsyncTruseraClient initialization."""
    client = AsyncTruseraClient(
        api_key="tsk_test_key",
        base_url="https://api.test.trusera.dev",
    )

    assert client.api_key == "tsk_test_key"
    assert client.base_url == "https://api.test.trusera.dev"

    await client.close()


@pytest.mark.asyncio
async def test_async_client_requires_api_key(mock_httpx_client):
    """Test that async client raises ValueError without API key."""
    with pytest.raises(ValueError, match="API key is required"):
        AsyncTruseraClient()


@pytest.mark.asyncio
async def test_async_client_env_var_fallback(mock_httpx_client, monkeypatch):
    """Test env var fallback for async client."""
    monkeypatch.setenv("TRUSERA_API_KEY", "tsk_async_env")

    client = AsyncTruseraClient()
    assert client.api_key == "tsk_async_env"

    await client.close()


@pytest.mark.asyncio
async def test_async_client_track_and_flush(mock_httpx_client):
    """Test async client event tracking and flushing."""
    client = AsyncTruseraClient(api_key="tsk_test_key")
    client.set_agent_id("agent_async_123")

    event = Event(type=EventType.TOOL_CALL, name="async_test")
    client.track(event)

    assert len(client._events) == 1

    await client.flush()

    assert len(client._events) == 0

    await client.close()


@pytest.mark.asyncio
async def test_async_client_context_manager(mock_httpx_client):
    """Test async client as context manager."""
    async with AsyncTruseraClient(api_key="tsk_test_key") as client:
        client.set_agent_id("agent_test")
        client.track(Event(type=EventType.TOOL_CALL, name="test"))

    assert client._closed is True


@pytest.mark.asyncio
async def test_async_client_track_after_close(mock_httpx_client, caplog):
    """Test that async client rejects events after close."""
    client = AsyncTruseraClient(api_key="tsk_test_key")
    await client.close()

    client.track(Event(type=EventType.TOOL_CALL, name="test"))
    assert "closed" in caplog.text.lower()


def test_flush_empty_queue(trusera_client, mock_httpx_client):
    """Test that flush with empty queue is a no-op (no API call)."""
    initial_call_count = mock_httpx_client.post.call_count

    trusera_client.flush()

    # No new post calls should have been made
    assert mock_httpx_client.post.call_count == initial_call_count


def test_register_agent_http_error(mock_httpx_client):
    """Test register_agent raises on HTTP error."""
    import httpx as _httpx

    mock_httpx_client.post.side_effect = _httpx.ConnectError("connection refused")

    client = TruseraClient(api_key="tsk_test_key", flush_interval=60.0)

    with pytest.raises(_httpx.ConnectError):
        client.register_agent("agent", "langchain")

    client.close()


def test_client_default_base_url(mock_httpx_client):
    """Test that default base URL is used when not provided."""
    client = TruseraClient(api_key="tsk_test_key")

    assert client.base_url == "https://api.trusera.dev"

    client.close()


def test_client_base_url_trailing_slash(mock_httpx_client):
    """Test that trailing slash is stripped from base URL."""
    client = TruseraClient(api_key="tsk_test_key", base_url="https://api.test.dev/")

    assert client.base_url == "https://api.test.dev"

    client.close()


# --- Additional AsyncTruseraClient tests ---


@pytest.mark.asyncio
async def test_async_flush_without_agent_id(mock_httpx_client, caplog):
    """Test that async flush warns when no agent ID set."""
    client = AsyncTruseraClient(api_key="tsk_test_key")
    client.track(Event(type=EventType.TOOL_CALL, name="test"))

    await client.flush()

    assert "No agent ID set" in caplog.text
    await client.close()


@pytest.mark.asyncio
async def test_async_close_idempotent(mock_httpx_client):
    """Test that async close can be called multiple times safely."""
    client = AsyncTruseraClient(api_key="tsk_test_key")
    await client.close()
    await client.close()  # Should not raise


@pytest.mark.asyncio
async def test_async_register_agent(mock_httpx_client):
    """Test async register_agent success path."""
    client = AsyncTruseraClient(api_key="tsk_test_key")

    agent_id = await client.register_agent("test-agent", "crewai")

    assert agent_id == "agent_123"
    assert client._agent_id == "agent_123"

    await client.close()


@pytest.mark.asyncio
async def test_async_flush_batch_size(mock_httpx_client):
    """Test async flush respects batch_size."""
    client = AsyncTruseraClient(api_key="tsk_test_key", batch_size=3)
    client.set_agent_id("agent_test")

    for i in range(5):
        client.track(Event(type=EventType.TOOL_CALL, name=f"tool_{i}"))

    await client.flush()

    # Should have 2 remaining (5 - batch_size of 3)
    assert len(client._events) == 2

    await client.close()
