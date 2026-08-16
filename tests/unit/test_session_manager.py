"""单元测试：覆盖 SessionManager 的会话缓存、清理和降级分支。"""

import asyncio
from datetime import datetime, timedelta

import pytest

from session_manager import SessionManager


pytestmark = [pytest.mark.unit, pytest.mark.full]


class FakeAgent:
    """用于验证会话清理时是否调用 Agent.cleanup。"""
    def __init__(self, fail_cleanup: bool = False):
        self.fail_cleanup = fail_cleanup
        self.cleaned = False

    async def cleanup(self):
        self.cleaned = True
        if self.fail_cleanup:
            raise RuntimeError("cleanup failed")


class FakeMCP:
    """用于验证全局 cleanup 会断开共享 MCP 客户端。"""
    def __init__(self):
        self.disconnected = False

    async def disconnect(self):
        self.disconnected = True


@pytest.mark.asyncio
async def test_get_or_create_agent_reuses_cached_instance(monkeypatch):
    """验证首次创建 Agent 后，同一 user/session 会复用缓存实例并刷新活跃时间。"""
    manager = SessionManager()
    created = []

    async def fake_initialize(user_id, session_id, external_urls, local_url, skills_info=""):
        created.append((user_id, session_id, tuple(external_urls), local_url, skills_info))
        return FakeAgent()

    monkeypatch.setattr(manager, "_initialize_agent", fake_initialize)

    first = await manager.get_or_create_agent("user", "session", ["ext"], "local", "skills")
    second = await manager.get_or_create_agent("user", "session", ["other"], "other-local")

    assert first is second
    assert created == [("user", "session", ("ext",), "local", "skills")]
    assert manager.get_agent("user", "session") is first
    with pytest.raises(ValueError):
        manager.get_agent("user", "missing")


@pytest.mark.asyncio
async def test_get_or_create_agent_initializes_different_sessions_concurrently(monkeypatch):
    """验证不同会话冷启动使用独立初始化锁，不会被全局锁串行化。"""
    manager = SessionManager()
    running = 0
    max_running = 0

    async def fake_initialize(user_id, session_id, external_urls, local_url, skills_info=""):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        await asyncio.sleep(0.01)
        running -= 1
        return FakeAgent()

    monkeypatch.setattr(manager, "_initialize_agent", fake_initialize)

    first, second = await asyncio.gather(
        manager.get_or_create_agent("user-a", "session-a", [], "local"),
        manager.get_or_create_agent("user-b", "session-b", [], "local"),
    )

    assert first is not second
    assert max_running == 2


@pytest.mark.asyncio
async def test_cleanup_session_handles_missing_and_agent_cleanup_errors():
    """验证清理不存在会话是空操作，Agent 清理异常不会阻断删除缓存。"""
    manager = SessionManager()
    failing = FakeAgent(fail_cleanup=True)
    manager.sessions[("user", "session")] = (failing, datetime.now())

    await manager.cleanup_session("user", "missing")
    await manager.cleanup_session("user", "session")

    assert failing.cleaned is True
    assert ("user", "session") not in manager.sessions


@pytest.mark.asyncio
async def test_cleanup_disconnects_all_agents_and_shared_mcp():
    """验证全局清理会清空所有会话并断开共享 MCP 连接。"""
    manager = SessionManager()
    first = FakeAgent()
    second = FakeAgent(fail_cleanup=True)
    mcp = FakeMCP()
    manager.sessions[("u1", "s1")] = (first, datetime.now())
    manager.sessions[("u2", "s2")] = (second, datetime.now())
    manager._mcp_client = mcp
    manager._mcp_local_ready = True

    await manager.cleanup()

    assert manager.sessions == {}
    assert first.cleaned is True
    assert second.cleaned is True
    assert mcp.disconnected is True
    assert manager._mcp_client is None
    assert manager._mcp_local_ready is False


@pytest.mark.asyncio
async def test_warmup_mcp_swallows_connection_errors(monkeypatch):
    """验证 MCP 预热失败只记录日志，不影响调用方继续启动。"""
    manager = SessionManager()

    async def fail_connect(*args, **kwargs):
        raise RuntimeError("mcp unavailable")

    monkeypatch.setattr(manager, "_get_or_create_mcp_client", fail_connect)

    await manager.warmup_mcp(["external"], "local")


def test_cleanup_expired_sessions_schedules_only_expired(monkeypatch):
    """验证超时清理只调度过期会话，未过期会话保持不动。"""
    manager = SessionManager()
    now = datetime.now()
    manager.sessions[("old-user", "old-session")] = (
        FakeAgent(),
        now - manager.session_timeout - timedelta(seconds=1),
    )
    manager.sessions[("new-user", "new-session")] = (FakeAgent(), now)
    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()
        return None

    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    manager.cleanup_expired_sessions()

    assert len(scheduled) == 1
