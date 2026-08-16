"""集成测试：覆盖 MCPClient 的本地 SSE 连接、工具描述缓存和缺失工具错误。"""

from types import SimpleNamespace

import asyncio
import httpx
import pytest

from mcp_client import MCPClient


pytestmark = [pytest.mark.integration, pytest.mark.full]


class FakeSession:
    """模拟 MCP session 的 call_tool 行为，用于直接测试客户端分发逻辑。"""
    async def call_tool(self, tool, args):
        return SimpleNamespace(content=[SimpleNamespace(text="搜索结果")])


class ClosingStream:
    """用于验证 MCP 断开时会关闭读写流。"""
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class ClosingStack:
    """用于验证 MCP 断开时会关闭资源栈。"""
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class AsyncContext:
    """简易异步上下文管理器，用于替代 SSE 和 MCP session。"""
    def __init__(self, value=None, exc: Exception | None = None):
        self.value = value
        self.exc = exc

    async def __aenter__(self):
        if self.exc:
            raise self.exc
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_mcp_client_connects_to_local_sse_and_calls_tools(local_mcp_url):
    """验证 MCPClient 能连接本地 SSE 服务并调用注册工具。"""
    client = MCPClient()
    try:
        assert await client.connect(local_mcp_url)
        assert "add_numbers" in client.tools
        assert "summary_session" in client.tools

        added = await client.call_tool("add_numbers", {"a": 2, "b": 3})
        summary = await client.call_tool("summary_session", {"conversation_history": "用户: hi"})

        assert "5" in added
        assert "本地 MCP 摘要" in summary
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_description_cache_and_direct_session_call():
    """验证工具描述缓存稳定，并能通过已绑定 session 直接调用工具。"""
    client = MCPClient()
    fake_tool = SimpleNamespace(
        name="fake_search",
        description="模拟外部搜索",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "关键词"}},
            "required": ["query"],
        },
    )
    session = FakeSession()
    client.tools["fake_search"] = fake_tool
    client._tool_to_sse_session["fake_search"] = session
    client._sse_session_to_url[session] = "memory://fake"
    client._connection_healthy["memory://fake"] = True

    first = client.get_tools_description()
    second = client.get_tools_description()
    called = await client.call_tool("fake_search", {"query": "上海"})

    assert first == second
    assert "fake_search" in first
    assert called == "搜索结果"


@pytest.mark.asyncio
async def test_mcp_client_reports_missing_tool():
    """验证调用不存在工具时抛出包含中文提示的运行时错误。"""
    client = MCPClient()
    client.tools["known"] = SimpleNamespace(
        name="known",
        description="known",
        inputSchema={},
    )

    with pytest.raises(RuntimeError, match="未找到对应"):
        await client.call_tool("missing", {})


@pytest.mark.asyncio
async def test_mcp_client_cached_connect_connect_all_extract_and_disconnect(monkeypatch):
    """验证 MCPClient 的缓存连接、批量连接、结果兜底提取和资源清理分支。"""
    client = MCPClient()
    calls = []

    async def fake_connect(url):
        calls.append(url)
        return url != "bad"

    monkeypatch.setattr(client, "connect", fake_connect)
    await client.connect_all(["ok", "bad", "also-ok"])

    assert calls == ["ok", "bad", "also-ok"]
    assert client._extract_result("plain") == "plain"
    assert "content=" in client._extract_result(SimpleNamespace(content=[]))

    read_stream = ClosingStream()
    write_stream = ClosingStream()
    stack = ClosingStack()
    session = SimpleNamespace(_read_stream=read_stream, _write_stream=write_stream)
    client._background_tasks["url"] = asyncio.create_task(asyncio.sleep(10))
    client._sse_sessions["url"] = session
    client._stacks["url"] = stack
    client.tools["known"] = SimpleNamespace(name="known", description="desc", inputSchema={})
    client._tools_description_cache = "cached"

    await client.disconnect()

    assert read_stream.closed is True
    assert write_stream.closed is True
    assert stack.closed is True
    assert client.tools == {}
    assert client.tools_version > 0


@pytest.mark.asyncio
async def test_mcp_client_call_tool_reconnects_unhealthy_session(monkeypatch):
    """验证连接不健康时，调用工具前会先重连并使用新 session。"""
    client = MCPClient()

    class OldSession:
        pass

    old_session = OldSession()
    new_session = FakeSession()
    tool = SimpleNamespace(name="lookup", description="lookup", inputSchema={})
    client.tools["lookup"] = tool
    client._tool_to_sse_session["lookup"] = old_session
    client._sse_session_to_url[old_session] = "memory://mcp"
    client._connection_healthy["memory://mcp"] = False

    async def fake_reconnect(session):
        assert session is old_session
        client._sse_session_to_url.pop(old_session, None)
        client._sse_session_to_url[new_session] = "memory://mcp"
        client._tool_to_sse_session["lookup"] = new_session
        return new_session

    monkeypatch.setattr(client, "_reconnect_sse_session", fake_reconnect)

    assert await client.call_tool("lookup", {"q": "x"}) == "搜索结果"


@pytest.mark.asyncio
async def test_mcp_client_call_tool_retries_after_transport_error(monkeypatch):
    """验证工具调用遇到传输错误时会重连并重试一次。"""
    client = MCPClient()

    class FailingSession:
        async def call_tool(self, tool, args):
            return SimpleNamespace(content=[SimpleNamespace(text="旧连接不应成功")])

    failing_session = FailingSession()
    successful_session = FakeSession()
    client.tools["lookup"] = SimpleNamespace(name="lookup", description="lookup", inputSchema={})
    client._tool_to_sse_session["lookup"] = failing_session
    client._sse_session_to_url[failing_session] = "memory://mcp"
    client._connection_healthy["memory://mcp"] = True
    attempts = {"count": 0}

    async def fake_wait_for(awaitable, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise httpx.ReadError("broken")
        return await awaitable

    async def fake_reconnect(session):
        client._sse_session_to_url.pop(failing_session, None)
        client._sse_session_to_url[successful_session] = "memory://mcp"
        client._tool_to_sse_session["lookup"] = successful_session
        return successful_session

    monkeypatch.setattr("mcp_client.asyncio.wait_for", fake_wait_for)
    monkeypatch.setattr(client, "_reconnect_sse_session", fake_reconnect)

    assert await client.call_tool("lookup", {"q": "x"}) == "搜索结果"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_mcp_client_connect_internal_success_replaces_url_tools_and_failure(monkeypatch):
    """验证 MCP 内部连接会注册工具、替换旧 URL 工具，并在连接失败时返回 False。"""
    client = MCPClient()
    old_session = object()
    client.tools["old_tool"] = SimpleNamespace(name="old_tool", description="old", inputSchema={})
    client._tool_to_sse_session["old_tool"] = old_session
    client._url_tools["memory://ok"] = {"old_tool"}

    class FakeClientSession:
        def __init__(self, read_stream, write_stream):
            self.read_stream = read_stream
            self.write_stream = write_stream

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="new_tool",
                        description="新工具",
                        inputSchema={"properties": {}},
                    )
                ]
            )

    monkeypatch.setattr("mcp_client.sse_client", lambda url: AsyncContext(("read", "write")))
    monkeypatch.setattr("mcp_client.ClientSession", FakeClientSession)

    assert await client._connect_internal("memory://ok")
    assert "old_tool" not in client.tools
    assert "new_tool" in client.tools
    assert client._connection_healthy["memory://ok"] is True
    assert client._background_tasks["memory://ok"] is not None

    await client.disconnect()

    monkeypatch.setattr("mcp_client.sse_client", lambda url: AsyncContext(exc=RuntimeError("connect failed")))
    assert not await client._connect_internal("http://127.0.0.1:1/sse")


@pytest.mark.asyncio
async def test_mcp_client_reconnect_cleans_old_resources_and_rebinds_tools(monkeypatch):
    """验证 MCP 重连会取消旧监控、关闭旧资源栈，并把工具绑定到新 session。"""
    client = MCPClient()
    old_session = object()
    new_session = object()
    old_stack = ClosingStack()
    monitor_task = asyncio.create_task(asyncio.sleep(10))
    client._background_tasks["memory://reconnect"] = monitor_task
    client._stacks["memory://reconnect"] = old_stack
    client._sse_sessions["memory://reconnect"] = old_session
    client._sse_session_to_url[old_session] = "memory://reconnect"
    client._tool_to_sse_session["lookup"] = old_session

    async def fake_connect_internal(url):
        client._sse_sessions[url] = new_session
        client._sse_session_to_url[new_session] = url
        return True

    monkeypatch.setattr(client, "_connect_internal", fake_connect_internal)

    rebound = await client._reconnect_sse_session(old_session)

    assert rebound is new_session
    assert old_stack.closed is True
    assert client._tool_to_sse_session["lookup"] is new_session
    assert old_session not in client._sse_session_to_url
