"""CrewAI integration for Trusera."""

import logging
from typing import Any, Optional

from ..client import TruseraClient
from ..events import Event, EventType

logger = logging.getLogger(__name__)

try:
    from crewai import Agent, Task

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


class TruseraCrewCallback:
    """
    CrewAI callback that sends events to Trusera.

    Captures task execution, agent actions, tool usage, and delegation.

    Example:
        >>> from crewai import Crew, Agent, Task
        >>> client = TruseraClient(api_key="tsk_...")
        >>> client.register_agent("my-crew", "crewai")
        >>> callback = TruseraCrewCallback(client)
        >>> crew = Crew(
        ...     agents=[agent],
        ...     tasks=[task],
        ...     step_callback=callback.step_callback
        ... )
        >>> crew.kickoff()
    """

    def __init__(self, client: TruseraClient) -> None:
        """
        Initialize the callback.

        Args:
            client: TruseraClient instance
        """
        self.client = client

    def step_callback(self, step_output: Any) -> None:
        """
        Callback for each step in the crew execution.

        Args:
            step_output: Output from the step
        """
        try:
            if hasattr(step_output, "task"):
                self._track_task(step_output)
            elif hasattr(step_output, "action"):
                self._track_action(step_output)
            else:
                # Generic step tracking
                event = Event(
                    type=EventType.DECISION,
                    name="crew_step",
                    payload={"output": str(step_output)[:500]},
                )
                self.client.track(event)
        except Exception as e:
            logger.error(f"Error in TruseraCrewCallback: {e}")

    def _get_task_name(self, task: Any) -> str:
        """Extract task name, with truncation indicator if needed."""
        desc = getattr(task, "description", None) or ""
        if len(desc) > 50:
            return desc[:50] + "..."
        return desc or "unknown_task"

    def _get_agent_role(self, agent: Any) -> str:
        """Safely extract agent role."""
        return getattr(agent, "role", None) or "unknown_agent"

    def _track_task(self, step_output: Any) -> None:
        """Track task execution."""
        task = getattr(step_output, "task", None)
        agent = getattr(step_output, "agent", None)

        task_name = self._get_task_name(task) if task else "unknown_task"
        task_desc = getattr(task, "description", "") or "" if task else ""
        agent_role = self._get_agent_role(agent) if agent else "unknown_agent"
        output = str(getattr(step_output, "output", ""))

        event = Event(
            type=EventType.DECISION,
            name=f"task_{task_name}",
            payload={
                "task_description": task_desc,
                "output": output[:500],
            },
            metadata={
                "agent_role": agent_role,
                "task_type": "crew_task",
            },
        )

        self.client.track(event)

    def _track_action(self, step_output: Any) -> None:
        """Track agent action."""
        action = getattr(step_output, "action", None)

        action_name = getattr(action, "tool", "unknown_action") if action else "unknown_action"
        action_input = getattr(action, "tool_input", {}) if action else {}
        output = str(getattr(step_output, "output", ""))

        # Detect delegation
        is_delegation = action_name in ("Delegate work to co-worker", "Ask question to co-worker")

        event_type = EventType.DECISION if is_delegation else EventType.TOOL_CALL
        metadata: dict[str, Any] = {"action_type": "crew_action"}

        if is_delegation:
            metadata["is_delegation"] = True
            # Try to extract co-worker name from input
            coworker = None
            if isinstance(action_input, dict):
                coworker = action_input.get("coworker") or action_input.get("co-worker")
            elif isinstance(action_input, str) and "coworker" in action_input.lower():
                coworker = action_input
            if coworker:
                metadata["delegated_to"] = str(coworker)

        event = Event(
            type=event_type,
            name=action_name,
            payload={
                "input": action_input if isinstance(action_input, (dict, str)) else str(action_input),
                "output": output[:500],
            },
            metadata=metadata,
        )

        self.client.track(event)

    def task_callback(self, task_output: Any) -> None:
        """
        Optional callback for task completion.

        Args:
            task_output: Output from the completed task
        """
        try:
            task = getattr(task_output, "task", None)
            task_name = self._get_task_name(task) if task else "unknown_task"
            output = str(getattr(task_output, "output", ""))

            event = Event(
                type=EventType.DECISION,
                name=f"task_complete_{task_name}",
                payload={
                    "output": output[:500],
                },
                metadata={
                    "task_status": "completed",
                },
            )

            self.client.track(event)
        except Exception as e:
            logger.error(f"Error in task_callback: {e}")
