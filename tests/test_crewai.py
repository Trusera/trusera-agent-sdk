"""Tests for CrewAI integration."""

from unittest.mock import Mock

from trusera_sdk import EventType
from trusera_sdk.integrations.crewai import TruseraCrewCallback


def test_callback_initialization(trusera_client):
    """Test TruseraCrewCallback initialization."""
    callback = TruseraCrewCallback(trusera_client)
    assert callback.client == trusera_client


def test_step_callback_task(trusera_client):
    """Test step_callback with a task step."""
    callback = TruseraCrewCallback(trusera_client)

    step = Mock()
    step.task = Mock()
    step.task.description = "Research AI trends"
    step.agent = Mock()
    step.agent.role = "Researcher"
    step.output = "Found 3 trends"

    callback.step_callback(step)

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.DECISION
    assert "Research AI trends" in event.name
    assert event.payload["task_description"] == "Research AI trends"
    assert event.metadata["agent_role"] == "Researcher"


def test_step_callback_action(trusera_client):
    """Test step_callback with an action step."""
    callback = TruseraCrewCallback(trusera_client)

    step = Mock(spec=["action", "output"])
    step.action = Mock()
    step.action.tool = "web_search"
    step.action.tool_input = {"query": "AI security"}
    step.output = "Search results here"

    callback.step_callback(step)

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.TOOL_CALL
    assert event.name == "web_search"
    assert event.payload["input"] == {"query": "AI security"}


def test_step_callback_generic(trusera_client):
    """Test step_callback with unknown step type."""
    callback = TruseraCrewCallback(trusera_client)

    step = Mock(spec=["output"])
    step.output = "generic output"

    callback.step_callback(step)

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.DECISION
    assert event.name == "crew_step"


def test_step_callback_delegation(trusera_client):
    """Test delegation tracking in step_callback."""
    callback = TruseraCrewCallback(trusera_client)

    step = Mock(spec=["action", "output"])
    step.action = Mock()
    step.action.tool = "Delegate work to co-worker"
    step.action.tool_input = {"coworker": "Writer", "task": "Write article"}
    step.output = "Delegated successfully"

    callback.step_callback(step)

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.DECISION
    assert event.metadata["is_delegation"] is True
    assert event.metadata["delegated_to"] == "Writer"


def test_task_callback(trusera_client):
    """Test task_callback for task completion."""
    callback = TruseraCrewCallback(trusera_client)

    task_output = Mock()
    task_output.task = Mock()
    task_output.task.description = "Write a blog post"
    task_output.output = "Blog post content here"

    callback.task_callback(task_output)

    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()
    assert event.type == EventType.DECISION
    assert "task_complete_" in event.name
    assert event.metadata["task_status"] == "completed"


def test_task_name_truncation(trusera_client):
    """Test that long task names are truncated with indicator."""
    callback = TruseraCrewCallback(trusera_client)
    long_desc = "A" * 100

    task = Mock()
    task.description = long_desc

    name = callback._get_task_name(task)
    assert len(name) == 53  # 50 chars + "..."
    assert name.endswith("...")


def test_task_name_short(trusera_client):
    """Test that short task names are not truncated."""
    callback = TruseraCrewCallback(trusera_client)

    task = Mock()
    task.description = "Short task"

    name = callback._get_task_name(task)
    assert name == "Short task"


def test_missing_attributes(trusera_client):
    """Test graceful handling of missing attributes."""
    callback = TruseraCrewCallback(trusera_client)

    # Step with task but no agent
    step = Mock()
    step.task = Mock()
    step.task.description = "Test task"
    step.agent = None

    callback.step_callback(step)
    event = trusera_client._queue.get()
    assert event.metadata["agent_role"] == "unknown_agent"


def test_step_callback_error_handling(trusera_client, caplog):
    """Test that step_callback errors don't crash."""
    callback = TruseraCrewCallback(trusera_client)

    # Force an error by making step_output raise on attribute access
    step = Mock()
    step.task = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    # Should not raise
    callback.step_callback(step)


def test_action_with_no_tool_input(trusera_client):
    """Test action tracking when tool_input is missing."""
    callback = TruseraCrewCallback(trusera_client)

    step = Mock(spec=["action", "output"])
    step.action = Mock(spec=["tool"])
    step.action.tool = "calculator"
    step.output = "42"

    callback.step_callback(step)

    event = trusera_client._queue.get()
    assert event.name == "calculator"
    assert event.payload["input"] == {}
