"""Tests for LangChain integration."""

import pytest
from unittest.mock import Mock
from uuid import uuid4

from trusera_sdk import EventType

try:
    from trusera_sdk.integrations.langchain import TruseraCallbackHandler
    from langchain_core.outputs import LLMResult, Generation

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_callback_handler_initialization(trusera_client):
    """Test TruseraCallbackHandler initialization."""
    handler = TruseraCallbackHandler(trusera_client)
    assert handler.client == trusera_client


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_llm_start_end(trusera_client):
    """Test LLM invocation tracking."""
    handler = TruseraCallbackHandler(trusera_client)
    run_id = uuid4()

    # Simulate LLM start
    handler.on_llm_start(
        serialized={"name": "gpt-4"},
        prompts=["What is AI?"],
        run_id=run_id,
    )

    # Simulate LLM end
    generations = [[Generation(text="AI is artificial intelligence.")]]
    result = LLMResult(generations=generations)
    handler.on_llm_end(response=result, run_id=run_id)

    # Check event
    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()

    assert event.type == EventType.LLM_INVOKE
    assert event.name == "llm_gpt-4"
    assert event.payload["prompts"] == ["What is AI?"]
    assert event.payload["generations"][0] == "AI is artificial intelligence."
    assert event.payload["model"] == "gpt-4"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_llm_error(trusera_client):
    """Test LLM error tracking."""
    handler = TruseraCallbackHandler(trusera_client)
    run_id = uuid4()

    # Simulate LLM start
    handler.on_llm_start(
        serialized={"name": "gpt-4"},
        prompts=["test prompt"],
        run_id=run_id,
    )

    # Simulate LLM error
    error = RuntimeError("API rate limit exceeded")
    handler.on_llm_error(error=error, run_id=run_id)

    # Check event
    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()

    assert event.type == EventType.LLM_INVOKE
    assert event.name == "llm_gpt-4"
    assert event.payload["error"]["type"] == "RuntimeError"
    assert event.payload["error"]["message"] == "API rate limit exceeded"
    assert event.metadata["success"] is False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_tool_start_end(trusera_client):
    """Test tool call tracking."""
    handler = TruseraCallbackHandler(trusera_client)
    run_id = uuid4()

    # Simulate tool start
    handler.on_tool_start(
        serialized={"name": "search_tool"},
        input_str="search query",
        run_id=run_id,
        inputs={"query": "test query"},
    )

    # Simulate tool end
    handler.on_tool_end(output="search results", run_id=run_id)

    # Check event
    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()

    assert event.type == EventType.TOOL_CALL
    assert event.name == "search_tool"
    assert event.payload["input"] == "search query"
    assert event.payload["inputs"] == {"query": "test query"}
    assert event.payload["output"] == "search results"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_tool_error(trusera_client):
    """Test tool error tracking."""
    handler = TruseraCallbackHandler(trusera_client)
    run_id = uuid4()

    # Simulate tool start
    handler.on_tool_start(
        serialized={"name": "failing_tool"},
        input_str="test input",
        run_id=run_id,
    )

    # Simulate tool error
    error = ValueError("Tool failed")
    handler.on_tool_error(error=error, run_id=run_id)

    # Check event
    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()

    assert event.type == EventType.TOOL_CALL
    assert event.name == "failing_tool"
    assert event.payload["error"]["type"] == "ValueError"
    assert event.payload["error"]["message"] == "Tool failed"
    assert event.metadata["success"] is False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_chain_start_end(trusera_client):
    """Test chain execution tracking."""
    handler = TruseraCallbackHandler(trusera_client)
    run_id = uuid4()

    # Simulate chain start
    handler.on_chain_start(
        serialized={"name": "qa_chain"},
        inputs={"question": "What is AI?"},
        run_id=run_id,
    )

    # Simulate chain end
    handler.on_chain_end(
        outputs={"answer": "AI is artificial intelligence."},
        run_id=run_id,
    )

    # Check event
    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()

    assert event.type == EventType.DECISION
    assert event.name == "chain_qa_chain"
    assert event.payload["inputs"] == {"question": "What is AI?"}
    assert event.payload["outputs"] == {"answer": "AI is artificial intelligence."}


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_retriever_start_end(trusera_client):
    """Test retriever tracking for RAG chains."""
    handler = TruseraCallbackHandler(trusera_client)
    run_id = uuid4()

    # Simulate retriever start
    handler.on_retriever_start(
        serialized={"name": "vector_store"},
        query="What is AI security?",
        run_id=run_id,
    )

    # Mock documents
    doc = Mock()
    doc.page_content = "AI security involves protecting AI systems."
    doc.metadata = {"source": "wiki"}

    handler.on_retriever_end(documents=[doc], run_id=run_id)

    # Check event
    assert trusera_client._queue.qsize() == 1
    event = trusera_client._queue.get()

    assert event.type == EventType.DATA_ACCESS
    assert event.name == "retriever_vector_store"
    assert event.payload["query"] == "What is AI security?"
    assert event.payload["num_documents"] == 1
    assert event.payload["documents"][0]["page_content"] == "AI security involves protecting AI systems."
    assert event.payload["documents"][0]["metadata"] == {"source": "wiki"}


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_multiple_events(trusera_client):
    """Test tracking multiple events."""
    handler = TruseraCallbackHandler(trusera_client)

    # LLM call
    run_id_1 = uuid4()
    handler.on_llm_start(serialized={"name": "gpt-4"}, prompts=["test"], run_id=run_id_1)
    handler.on_llm_end(
        response=LLMResult(generations=[[Generation(text="response")]]),
        run_id=run_id_1,
    )

    # Tool call
    run_id_2 = uuid4()
    handler.on_tool_start(serialized={"name": "tool"}, input_str="input", run_id=run_id_2)
    handler.on_tool_end(output="output", run_id=run_id_2)

    # Should have 2 events
    assert trusera_client._queue.qsize() == 2


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_metadata_ttl_cleanup(trusera_client):
    """Test that stale metadata entries are cleaned up."""
    import time
    from trusera_sdk.integrations import langchain as lc_mod

    handler = TruseraCallbackHandler(trusera_client)

    # Store an entry and manually age it
    handler._store_metadata("old_run", {"data": "old"})
    handler._run_metadata["old_run"]["_stored_at"] = time.monotonic() - (lc_mod._METADATA_TTL_SECONDS + 1)

    # Store a new entry — triggers cleanup
    handler._store_metadata("new_run", {"data": "new"})

    assert "old_run" not in handler._run_metadata
    assert "new_run" in handler._run_metadata
