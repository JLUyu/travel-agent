"""集成测试：覆盖 SessionManager 与 MCP/Agent 生命周期的协作路径。"""

import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from session_manager import SessionManager


pytestmark = [pytest.mark.integration, pytest.mark.full]


class ManagedAgent:
    """用于验证 SessionManager 生命周期管理的 Agent 替身。"""
    def __init__(self):
        self.cleaned = False

    async def cleanup(self):
        self.cleaned = True


@pytest.mark.asyncio
async def test_session_manager_mcp_client_local_success_and_failure(monkeypatch):
    """验证共享 MCP 客户端会复用，并在本地 MCP 连接失败时抛出明确错误。"""
    created = []

    class FakeMCPClient:
        connect_result = True

        def __init__(self):
            self.external_urls = []
            self.local_urls = []
            created.append(self)

        async def connect_all(self, urls):
            self.external_urls.append(tuple(urls))

        async def connect(self, url):
            self.local_urls.append(url)
            return self.connect_result

    monkeypatch.setitem(sys.modules, "mcp_client", SimpleNamespace(MCPClient=FakeMCPClient))
    manager = SessionManager()

    first = await manager._get_or_create_mcp_client(["a", "b"], "local", require_local=True)
    second = await manager._get_or_create_mcp_client(["c"], "local", require_local=True)

    assert first is second
    assert len(created) == 1
    assert first.external_urls == [("a", "b"), ("c",)]
    assert first.local_urls == ["local"]

    failing_manager = SessionManager()
    FakeMCPClient.connect_result = False
    with pytest.raises(RuntimeError, match="local MCP"):
        await failing_manager._get_or_create_mcp_client([], "local", require_local=True)


@pytest.mark.asyncio
async def test_session_manager_agent_cache_cleanup_and_expiry(monkeypatch):
    """验证 Agent 创建复用、过期调度和 cleanup 会协同清理 Agent。"""
    manager = SessionManager()
    created = []

    async def fake_initialize(*args, **kwargs):
        agent = ManagedAgent()
        created.append(agent)
        return agent

    monkeypatch.setattr(manager, "_initialize_agent", fake_initialize)

    first = await manager.get_or_create_agent("user", "session", [], "")
    second = await manager.get_or_create_agent("user", "session", [], "")

    assert first is second
    assert len(created) == 1

    manager.sessions[("expired", "session")] = (
        ManagedAgent(),
        datetime.now() - manager.session_timeout - timedelta(seconds=1),
    )
    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()
        return None

    monkeypatch.setattr("asyncio.create_task", fake_create_task)
    manager.cleanup_expired_sessions()

    assert len(scheduled) == 1

    await manager.cleanup_session("user", "session")
    assert first.cleaned is True

    mcp = SimpleNamespace(disconnected=False)

    async def disconnect():
        mcp.disconnected = True

    mcp.disconnect = disconnect
    remaining = ManagedAgent()
    manager.sessions[("left", "session")] = (remaining, datetime.now())
    manager._mcp_client = mcp
    manager._mcp_local_ready = True

    await manager.cleanup()

    assert remaining.cleaned is True
    assert mcp.disconnected is True
    assert manager.sessions == {}
