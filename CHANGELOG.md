# Changelog

All notable changes to the Trusera Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-14

### Added
- `AsyncTruseraClient` for asyncio-native applications
- Environment variable fallback (`TRUSERA_API_KEY`, `TRUSERA_API_URL`)
- Configurable `max_retries` for flush — events dropped after exhaustion
- Thread-safe flush via `_flush_lock`
- LangChain: `on_llm_error` handler, `on_retriever_start/end` for RAG chains
- LangChain: metadata TTL cleanup to prevent memory leaks
- CrewAI: delegation tracking (`Delegate work to co-worker`)
- CrewAI: task name truncation indicator
- AutoGen: modern `tool_calls` format support (alongside legacy `function_call`)
- AutoGen: `functools.wraps` in `function_hook` (preserves metadata)
- Decorator: extended `_serialize_value` — bytes, set, datetime, date, Enum
- Decorator: payload size truncation at 64 KB
- Event: `__repr__` for debugging, `from_dict()` validation
- Comprehensive test suite: 95 tests across 6 files

### Changed
- `api_key` is now `Optional` — falls back to `TRUSERA_API_KEY` env var
- User-Agent header uses dynamic version from `importlib.metadata`
- PEP 561: `py.typed` moved into `trusera_sdk/` package
- AutoGen/CrewAI integrations no longer require framework installed at import

### Fixed
- Circular import when reading `__version__` from client module
- Potential OOM from infinite re-queuing on persistent API failures
- Race condition in concurrent flush calls
- Missing error events in LangChain LLM error path

## [0.1.0] - 2026-02-13

### Added
- Initial release of the Trusera Python SDK
- Core `TruseraClient` for event tracking and API communication
- Event types: `TOOL_CALL`, `LLM_INVOKE`, `DATA_ACCESS`, `API_CALL`, `FILE_WRITE`, `DECISION`
- `@monitor` decorator for automatic function tracking
- Support for both sync and async functions
- Automatic batching and background flushing
- Context manager support for clean resource management
- LangChain integration with `TruseraCallbackHandler`
- CrewAI integration with `TruseraCrewCallback`
- AutoGen integration with `TruseraAutoGenHook`
- Comprehensive test suite with >90% coverage
- Type hints throughout the codebase
- Apache 2.0 license

### Framework Support
- LangChain Core >=0.1.0
- CrewAI >=0.1.0
- AutoGen >=0.2.0

### Development Tools
- pytest for testing
- pytest-asyncio for async test support
- ruff for linting
- mypy for type checking
- GitHub Actions for CI/CD

[0.2.0]: https://github.com/Trusera/trusera-agent-sdk/releases/tag/v0.2.0
[0.1.0]: https://github.com/Trusera/trusera-agent-sdk/releases/tag/v0.1.0
