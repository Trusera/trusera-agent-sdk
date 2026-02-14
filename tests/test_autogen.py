"""Tests for AutoGen integration."""

from unittest.mock import Mock

import pytest

from trusera_sdk import EventType
from trusera_sdk.integrations.autogen import TruseraAutoGenHook


def test_hook_initialization(trusera_client):
    """Test TruseraAutoGenHook initialization."""
    hook = TruseraAutoGenHook(trusera_client)
    assert hook.client == trusera_client


def test_message_hook_regular(trusera_client):
    """Test message_hook with a regular text message."""
    hook = TruseraAutoGenHook(trusera_client)

    sender = Mock()
    sender.name = "assistant"
    recipient = Mock()
    recipient.name = "user_proxy"
    message = {"content": "Hello, how can I help?"}

    hook.message_hook(sender, recipient, message)

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.DECISION
    assert event.name == "message_assistant_to_user_proxy"
    assert event.payload["content"] == "Hello, how can I help?"
    assert event.payload["sender"] == "assistant"
    assert event.payload["recipient"] == "user_proxy"
    assert event.metadata["message_type"] == "autogen_message"


def test_message_hook_function_call(trusera_client):
    """Test message_hook with legacy function_call format."""
    hook = TruseraAutoGenHook(trusera_client)

    sender = Mock()
    sender.name = "assistant"
    recipient = Mock()
    recipient.name = "executor"
    message = {
        "function_call": {
            "name": "search_web",
            "arguments": {"query": "AI security"},
        }
    }

    hook.message_hook(sender, recipient, message)

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.TOOL_CALL
    assert event.name == "search_web"
    assert event.payload["arguments"] == {"query": "AI security"}
    assert event.payload["sender"] == "assistant"
    assert event.payload["recipient"] == "executor"
    assert event.metadata["call_type"] == "autogen_function_call"


def test_message_hook_tool_calls(trusera_client):
    """Test message_hook with modern tool_calls format."""
    hook = TruseraAutoGenHook(trusera_client)

    sender = Mock()
    sender.name = "assistant"
    recipient = Mock()
    recipient.name = "executor"
    message = {
        "tool_calls": [
            {
                "id": "call_001",
                "function": {
                    "name": "calculate",
                    "arguments": {"expression": "2+2"},
                },
            }
        ]
    }

    hook.message_hook(sender, recipient, message)

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.TOOL_CALL
    assert event.name == "calculate"
    assert event.payload["arguments"] == {"expression": "2+2"}
    assert event.payload["tool_call_id"] == "call_001"
    assert event.metadata["call_type"] == "autogen_tool_call"


def test_message_hook_multiple_tool_calls(trusera_client):
    """Test message_hook with multiple tool_calls creates multiple events."""
    hook = TruseraAutoGenHook(trusera_client)

    sender = Mock()
    sender.name = "assistant"
    recipient = Mock()
    recipient.name = "executor"
    message = {
        "tool_calls": [
            {"id": "call_1", "function": {"name": "search", "arguments": {"q": "a"}}},
            {"id": "call_2", "function": {"name": "fetch", "arguments": {"url": "b"}}},
        ]
    }

    hook.message_hook(sender, recipient, message)

    assert trusera_client._queue.qsize() == 2
    event1 = trusera_client._queue.get()
    event2 = trusera_client._queue.get()
    assert event1.name == "search"
    assert event2.name == "fetch"


def test_message_hook_tool_calls_priority(trusera_client):
    """Test that tool_calls takes priority over function_call when both present."""
    hook = TruseraAutoGenHook(trusera_client)

    sender = Mock()
    sender.name = "assistant"
    recipient = Mock()
    recipient.name = "executor"
    message = {
        "tool_calls": [
            {"id": "call_1", "function": {"name": "modern_tool", "arguments": {}}},
        ],
        "function_call": {"name": "legacy_tool", "arguments": {}},
    }

    hook.message_hook(sender, recipient, message)

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.name == "modern_tool"
    assert event.metadata["call_type"] == "autogen_tool_call"


def test_function_hook_success(trusera_client):
    """Test function_hook tracks successful execution."""
    hook = TruseraAutoGenHook(trusera_client)

    def add(a, b):
        """Add two numbers."""
        return a + b

    wrapped = hook.function_hook(add)
    result = wrapped(2, 3)

    assert result == 5
    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.TOOL_CALL
    assert event.name == "add"
    assert event.payload["result"] == 5
    assert event.metadata["function_type"] == "autogen_function"
    assert event.metadata["success"] is True


def test_function_hook_error(trusera_client):
    """Test function_hook tracks errors and re-raises."""
    hook = TruseraAutoGenHook(trusera_client)

    def failing():
        raise ValueError("bad input")

    wrapped = hook.function_hook(failing)

    with pytest.raises(ValueError, match="bad input"):
        wrapped()

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.TOOL_CALL
    assert event.name == "failing"
    assert event.payload["error"]["type"] == "ValueError"
    assert event.payload["error"]["message"] == "bad input"
    assert event.metadata["success"] is False


def test_function_hook_preserves_metadata(trusera_client):
    """Test function_hook preserves function name and docstring via wraps."""
    hook = TruseraAutoGenHook(trusera_client)

    def documented_func(x: int) -> int:
        """Multiply by two."""
        return x * 2

    wrapped = hook.function_hook(documented_func)
    assert wrapped.__name__ == "documented_func"
    assert wrapped.__doc__ == "Multiply by two."


def test_function_hook_serializes_complex_args(trusera_client):
    """Test function_hook uses _serialize_value for args (not str)."""
    hook = TruseraAutoGenHook(trusera_client)

    def process(data):
        return len(data)

    wrapped = hook.function_hook(process)
    wrapped({"key": [1, 2, 3]})

    event = trusera_client._queue.get()
    args_payload = event.payload["arguments"]
    assert isinstance(args_payload, dict)
    # args tuple is serialized as list
    assert args_payload["args"][0] == {"key": [1, 2, 3]}


def test_setup_agent_with_hooks(trusera_client):
    """Test setup_agent registers hook on compatible agent."""
    hook = TruseraAutoGenHook(trusera_client)

    agent = Mock()
    agent.name = "test_agent"
    agent.register_hook = Mock()

    hook.setup_agent(agent)

    agent.register_hook.assert_called_once_with(
        "process_message_before_send", hook.message_hook
    )


def test_setup_agent_without_hooks(trusera_client, caplog):
    """Test setup_agent logs warning for incompatible agent."""
    hook = TruseraAutoGenHook(trusera_client)

    agent = Mock(spec=[])
    agent.name = "no_hooks_agent"

    hook.setup_agent(agent)

    # Should not raise — just logs a warning


def test_message_hook_error_handling(trusera_client):
    """Test message_hook handles errors gracefully without crashing."""
    hook = TruseraAutoGenHook(trusera_client)

    sender = Mock()
    sender.name = "test"
    recipient = Mock()
    recipient.name = "test"

    # Force error by making message.get raise
    message = Mock()
    message.get = Mock(side_effect=RuntimeError("boom"))

    # Should not raise
    hook.message_hook(sender, recipient, message)


def test_message_content_truncation(trusera_client):
    """Test that long message content is truncated to 500 chars."""
    hook = TruseraAutoGenHook(trusera_client)

    sender = Mock()
    sender.name = "assistant"
    recipient = Mock()
    recipient.name = "user"
    long_content = "x" * 1000
    message = {"content": long_content}

    hook.message_hook(sender, recipient, message)

    event = trusera_client._queue.get()
    assert len(event.payload["content"]) == 500


def test_missing_sender_name(trusera_client):
    """Test handling of sender/recipient without name attribute."""
    hook = TruseraAutoGenHook(trusera_client)

    sender = object()  # no name attribute
    recipient = object()
    message = {"content": "test"}

    hook.message_hook(sender, recipient, message)

    event = trusera_client._queue.get()
    assert event.name == "message_unknown_to_unknown"
