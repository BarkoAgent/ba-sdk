import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure env vars don't leak between tests."""
    monkeypatch.delenv("BACKEND_WS_URI", raising=False)
    monkeypatch.delenv("DEFAULT_WS_BASE", raising=False)
    monkeypatch.delenv("AGENT_CONNECTION_TYPE", raising=False)
    monkeypatch.delenv("CONCURRENCY_LIMIT", raising=False)
    monkeypatch.delenv("ENABLE_STREAMING", raising=False)
    monkeypatch.delenv("BARKO_RICH_BULK_OUTPUT", raising=False)
