"""AutoGen integration for Trusera."""

import functools
import logging
from typing import Any, Callable

from ..client import TruseraClient
from ..decorators import _serialize_value
from ..events import Event, EventType

logger = logging.getLogger(__name__)


class TruseraAutoGenHook:
    """
    AutoGen message hook that sends events to Trusera.

    Captures agent messages, tool calls, and function executions.
    Works with both legacy ``function_call`` and modern ``tool_calls``
    message formats.

    Example:
        >>> import autogen
        >>> client = TruseraClient(api_key="tsk_...")
        >>> client.register_agent("my-autogen", "autogen")
        >>> hook = TruseraAutoGenHook(client)
        >>> assistant = autogen.AssistantAgent(
        ...     name="assistant",
        ...     llm_config={"model": "gpt-4"},
        ... )
        >>> assistant.register_hook("process_message_before_send", hook.message_hook)
    """

    def __init__(self, client: TruseraClient) -> None:
        """
        Initialize the hook.

        Args:
            client: TruseraClient instance
        """
        self.client = client

    def message_hook(self, sender: Any, recipient: Any, message: dict[str, Any]) -> None:
        """
        Hook for AutoGen message processing.

        Handles regular messages, legacy ``function_call``, and modern
        ``tool_calls`` formats.

        Args:
            sender: The agent sending the message
            recipient: The agent receiving the message
            message: The message content
        """
        try:
            sender_name = getattr(sender, "name", "unknown")
            recipient_name = getattr(recipient, "name", "unknown")

            # Check for tool_calls (modern format, takes priority)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                self._track_tool_calls(sender_name, recipient_name, tool_calls)
                return

            # Check for function_call (legacy format)
            function_call = message.get("function_call")
            if function_call:
                self._track_function_call(sender_name, recipient_name, function_call)
                return

            # Track regular message
            content = message.get("content", "")

            event = Event(
                type=EventType.DECISION,
                name=f"message_{sender_name}_to_{recipient_name}",
                payload={
                    "content": str(content)[:500],
                    "sender": sender_name,
                    "recipient": recipient_name,
                },
                metadata={
                    "message_type": "autogen_message",
                },
            )

            self.client.track(event)

        except Exception as e:
            logger.error(f"Error in TruseraAutoGenHook: {e}")

    def _track_function_call(
        self, sender: str, recipient: str, function_call: dict[str, Any]
    ) -> None:
        """Track a legacy function_call."""
        function_name = function_call.get("name", "unknown_function")
        arguments = function_call.get("arguments", {})

        event = Event(
            type=EventType.TOOL_CALL,
            name=function_name,
            payload={
                "arguments": _serialize_value(arguments),
                "sender": sender,
                "recipient": recipient,
            },
            metadata={
                "call_type": "autogen_function_call",
            },
        )

        self.client.track(event)

    def _track_tool_calls(
        self, sender: str, recipient: str, tool_calls: list[dict[str, Any]]
    ) -> None:
        """Track modern tool_calls (one event per tool call)."""
        for call in tool_calls:
            function = call.get("function", {})
            call_id = call.get("id", "")
            function_name = function.get("name", "unknown_function")
            arguments = function.get("arguments", {})

            event = Event(
                type=EventType.TOOL_CALL,
                name=function_name,
                payload={
                    "arguments": _serialize_value(arguments),
                    "sender": sender,
                    "recipient": recipient,
                    "tool_call_id": call_id,
                },
                metadata={
                    "call_type": "autogen_tool_call",
                },
            )

            self.client.track(event)

    def function_hook(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator to track function executions in AutoGen.

        Preserves the original function's metadata (name, docstring, etc.)
        via ``functools.wraps``.

        Args:
            func: The function to wrap

        Returns:
            Wrapped function that tracks execution
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)

                event = Event(
                    type=EventType.TOOL_CALL,
                    name=func.__name__,
                    payload={
                        "arguments": _serialize_value(
                            {"args": args, "kwargs": kwargs}
                        ),
                        "result": _serialize_value(result),
                    },
                    metadata={
                        "function_type": "autogen_function",
                        "success": True,
                    },
                )

                self.client.track(event)
                return result

            except Exception as e:
                event = Event(
                    type=EventType.TOOL_CALL,
                    name=func.__name__,
                    payload={
                        "arguments": _serialize_value(
                            {"args": args, "kwargs": kwargs}
                        ),
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                        },
                    },
                    metadata={
                        "function_type": "autogen_function",
                        "success": False,
                    },
                )

                self.client.track(event)
                raise

        return wrapper

    def setup_agent(self, agent: Any) -> None:
        """
        Setup hooks for an AutoGen agent.

        Args:
            agent: The AutoGen agent to instrument
        """
        if hasattr(agent, "register_hook"):
            agent.register_hook("process_message_before_send", self.message_hook)
            logger.info(f"Registered Trusera hook for agent: {getattr(agent, 'name', 'unknown')}")
        else:
            logger.warning(f"Agent {getattr(agent, 'name', 'unknown')} does not support hooks")
