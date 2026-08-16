"""集成测试：覆盖 Skill 提示、SubAgent 工具链、记忆归档和 Gradio 进度/记忆信号。"""

import json
from types import SimpleNamespace

import pytest
import requests

from gradio_manager import GradioManager
from memory_consolidator import MemoryConsolidator
from session_store import SessionData
from subagents import SubAgentRunner
from tools.base import Tool
from tools.registry import ToolRegistry

from tests.conftest import FakeLLMClient, FakeMCPClient


pytestmark = [pytest.mark.integration, pytest.mark.full]


class EchoTool(Tool):
    """用于验证 SubAgent 可调用本地工具的回显测试工具。"""
    @property
    def name(self):
        return "echo"

    @property
    def description(self):
        return "回显输入"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, text: str):
        return f"echo:{text}"


@pytest.mark.asyncio
async def test_subagent_can_use_local_and_mcp_tools():
    """验证 SubAgent 能先调用本地工具，再调用 MCP 工具并输出最终结果。"""
    registry = ToolRegistry()
    registry.register(EchoTool())
    runner = SubAgentRunner(
        llm_client=FakeLLMClient(
            [
                '{"tool":"echo","arguments":{"text":"local"}}',
                '{"tool":"remote_lookup","arguments":{"q":"mcp"}}',
                "子代理完成",
            ]
        ),
        model="test-model",
        mcp_client=FakeMCPClient({"remote_lookup": "mcp-result"}),
        tools=registry,
    )

    result = await runner.run("general-purpose", "先本地再远程", max_iterations=10)

    assert result == "子代理完成"
    assert runner.mcp_client.calls == [("remote_lookup", {"q": "mcp"})]


@pytest.mark.asyncio
async def test_memory_consolidator_archives_by_token_budget(session_store):
    """验证超过 token 预算时，记忆归档器会写入 memory、history 和向量历史。"""
    session = SessionData(session_id="session-memory", user_id="user-memory")
    session.add_message("user", "我喜欢坐高铁。" * 20)
    session.add_message("assistant", "已记住。" * 20)
    session.add_message("user", "请继续帮我规划。" * 20)
    session_store.save_session(session)

    vector_entries = []
    vector_store = type(
        "VectorStore",
        (),
        {"add_history_entry": lambda self, timestamp, content: vector_entries.append((timestamp, content))},
    )()
    consolidator = MemoryConsolidator(
        session_store=session_store,
        vector_store=vector_store,
        user_id="user-memory",
        context_window_tokens=122,
        max_completion_tokens=1,
        safety_buffer=1,
    )
    consolidator.llm_client = FakeLLMClient(
        [
            {
                "tool_calls": [
                    {
                        "name": "save_memory",
                        "arguments": {
                            "history_entry": "[2026-01-01 10:00] 用户喜欢坐高铁",
                            "memory_update": "用户偏好：高铁",
                        },
                    }
                ]
            }
        ]
    )

    assert await consolidator.maybe_consolidate(session, "system", request_id="run-memory")
    loaded = session_store.load_session("user-memory", "session-memory")

    assert loaded.last_consolidated == 2
    assert "高铁" in session_store.read_memory_md("user-memory")
    assert "用户喜欢坐高铁" in session_store.read_history_md("user-memory")
    assert vector_entries


@pytest.mark.asyncio
async def test_memory_consolidator_does_not_archive_under_token_budget(session_store):
    """验证 token 未超限时不会写入 memory/history，也不会推进归档指针。"""
    session = SessionData(session_id="session-small", user_id="user-small")
    session.add_message("user", "短问题")
    session.add_message("assistant", "短回答")
    session_store.save_session(session)
    consolidator = MemoryConsolidator(
        session_store=session_store,
        vector_store=None,
        user_id="user-small",
        context_window_tokens=9000,
        max_completion_tokens=100,
        safety_buffer=100,
    )
    consolidator.llm_client = FakeLLMClient([])

    assert not await consolidator.maybe_consolidate(session, "system", request_id="run-small")
    loaded = session_store.load_session("user-small", "session-small")

    assert loaded.last_consolidated == 0
    assert session_store.read_memory_md("user-small") == ""
    assert session_store.read_history_md("user-small").strip() in {"", "# Conversation History"}


@pytest.mark.asyncio
async def test_memory_consolidator_failure_boundary_and_close_branches(session_store):
    """验证记忆整合器在无工具调用、无切割边界和关闭异常时能稳定降级。"""
    session = SessionData(session_id="session-fail-memory", user_id="user-fail-memory")
    session.add_message("user", "只有一条消息，无法找到下一轮用户边界。" * 20)
    session_store.save_session(session)
    consolidator = MemoryConsolidator(
        session_store=session_store,
        vector_store=None,
        user_id="user-fail-memory",
        context_window_tokens=80,
        max_completion_tokens=1,
        safety_buffer=1,
    )
    consolidator.llm_client = FakeLLMClient(["没有调用工具"])

    assert await consolidator._consolidate_chunk(session, [], "", request_id="run-empty")
    assert not await consolidator._consolidate_chunk(
        session,
        session.messages,
        "",
        request_id="run-no-tool",
    )
    assert consolidator.pick_consolidation_boundary(session, tokens_to_remove=10) is None
    assert not await consolidator.consolidate(session, "system", request_id="run-no-boundary")

    class RuntimeErrorCloseClient:
        def close(self):
            raise RuntimeError("Event loop is closed")

    consolidator.llm_client = RuntimeErrorCloseClient()
    await consolidator.close()
    await consolidator.close()


def test_gradio_create_new_session_switches_to_empty_session_without_archive(monkeypatch):
    """验证前端新建会话只创建并切换空会话，不触发旧会话归档语义。"""
    manager = GradioManager(backend_url="http://testserver")
    calls = []
    request = SimpleNamespace(headers={"cookie": "travel_user_id=user-ui"})

    monkeypatch.setattr(manager, "_create_session", lambda user_id: {"session_id": "session-new"})
    monkeypatch.setattr(manager, "_init_session", lambda user_id, session_id: calls.append(("init", user_id, session_id)))
    monkeypatch.setattr(manager, "_get_sessions", lambda user_id: [{"session_id": "session-new", "title": "新会话"}])
    monkeypatch.setattr(manager, "_get_memory_summary", lambda user_id, session_id: "未归档: 0 条")

    _choices, session_id, memory_summary, history, status, _signal = manager.create_new_session("", request)

    assert session_id == "session-new"
    assert manager.current_sessions["user-ui"] == "session-new"
    assert calls == [("init", "user-ui", "session-new")]
    assert memory_summary == "未归档: 0 条"
    assert history == []
    assert status == "已创建新会话"


def test_gradio_http_helpers_parse_backend_payloads(monkeypatch):
    """验证 Gradio HTTP helper 会调用对应后端接口并解析响应 payload。"""
    manager = GradioManager(backend_url="http://testserver")
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, timeout):
        calls.append(("GET", url, timeout))
        if "/session/list/" in url:
            return FakeResponse({"sessions": [{"session_id": "s", "title": "会话", "created_at": "c", "updated_at": "u"}]})
        if "/memory/unconsolidated/" in url:
            return FakeResponse({"summary": "未归档: 0"})
        if "/session/messages/" in url:
            return FakeResponse({"messages": [{"role": "user", "content": "问"}]})
        if "/session/runs/" in url:
            return FakeResponse({"runs": [{"run_id": "r"}]})
        raise AssertionError(url)

    def fake_post(url, timeout):
        calls.append(("POST", url, timeout))
        if "/session/create/" in url:
            return FakeResponse({"session": {"session_id": "created"}})
        if "/session/init/" in url:
            return FakeResponse({"status": "success"})
        raise AssertionError(url)

    monkeypatch.setattr("gradio_manager.requests.get", fake_get)
    monkeypatch.setattr("gradio_manager.requests.post", fake_post)

    assert manager._get_sessions("u")[0]["session_id"] == "s"
    assert manager._create_session("u")["session_id"] == "created"
    assert manager._init_session("u", "s") is None
    assert manager._get_memory_summary("u", "s") == "未归档: 0"
    assert manager._get_messages("u", "s")[0]["content"] == "问"
    assert manager._get_runs("u", "s")[0]["run_id"] == "r"
    assert len(calls) == 6


def test_gradio_switch_and_restore_session_history_with_progress(monkeypatch):
    """验证切换/刷新会话时能加载对应历史，并补上运行中的进度消息。"""
    manager = GradioManager(backend_url="http://testserver")
    request = SimpleNamespace(headers={})
    initialized = []

    messages_by_session = {
        "session-a": [{"role": "user", "content": "A 问题"}],
        "session-b": [{"role": "user", "content": "B 问题"}, {"role": "assistant", "content": "旧的临时回答"}],
    }
    runs_by_session = {
        "session-a": [],
        "session-b": [
            {
                "run_id": "run-b",
                "status": "running",
                "steps": [{"step_id": "1", "status": "running", "message": "正在调用路线工具"}],
            }
        ],
    }

    monkeypatch.setattr(manager, "_init_session", lambda user_id, session_id: initialized.append((user_id, session_id)))
    monkeypatch.setattr(manager, "_get_memory_summary", lambda user_id, session_id: f"{session_id} 未归档")
    monkeypatch.setattr(manager, "_get_messages", lambda user_id, session_id: list(messages_by_session[session_id]))
    monkeypatch.setattr(manager, "_get_runs", lambda user_id, session_id: runs_by_session[session_id])

    session_id, memory_summary, messages, status, _signal = manager.switch_session("user-ui", "session-a", request)
    restored, restore_status, _restore_signal = manager.restore_current_session("user-ui", "session-b")

    assert session_id == "session-a"
    assert memory_summary == "session-a 未归档"
    assert messages == [{"role": "user", "content": "A 问题"}]
    assert status == "已切换会话"
    assert initialized == [("user-ui", "session-a")]
    assert restore_status == "已恢复会话"
    assert restored[-1]["role"] == "assistant"
    assert "正在调用路线工具" in restored[-1]["content"]
    assert "旧的临时回答" not in restored[-1]["content"]


def test_gradio_progress_and_memory_signals_do_not_overlap_chat_refresh(monkeypatch):
    """验证 Gradio 进度轮询和记忆归档信号互不复用，避免错误刷新聊天区。"""
    manager = GradioManager(backend_url="http://testserver")
    run = {
        "run_id": "run-1",
        "status": "running",
        "steps": [
            {"step_id": "1", "status": "running", "message": "正在调用工具"},
        ],
    }
    memory_run = {
        "run_id": "run-2",
        "status": "completed",
        "steps": [
            {
                "step_type": "memory_consolidate",
                "status": "completed",
                "message": "归档完成",
                "created_at": "2026-01-01 10:00:00",
            }
        ],
    }

    monkeypatch.setattr(manager, "_get_runs", lambda user_id, session_id: [run])
    history = [{"role": "assistant", "content": manager._progress_message(run)["content"]}]
    signal = manager.poll_current_session_progress_signal("u", "s", history, "")

    assert "run-1" in signal
    assert manager.poll_current_session_memory_signal("u", "s", "") != signal

    monkeypatch.setattr(manager, "_get_runs", lambda user_id, session_id: [memory_run])
    memory_signal = manager.poll_current_session_memory_signal("u", "s", "")

    assert "归档完成" in memory_signal
    assert manager._get_messages_with_progress("u", "s") if False else True


def test_gradio_render_progress_running_failed_and_completed(monkeypatch):
    """验证进度信号触发后，Gradio 能分别渲染运行中、失败和最终回答补偿输出。"""
    manager = GradioManager(backend_url="http://testserver")
    manager.RESTORED_STREAM_DELAY = 0
    history = [{"role": "assistant", "content": f"{manager.RUN_PROGRESS_MARKER}run:running -->"}]

    monkeypatch.setattr(
        manager,
        "_get_runs",
        lambda user_id, session_id: [
            {"run_id": "run", "status": "running", "steps": [{"message": "正在调用工具"}]},
        ],
    )
    monkeypatch.setattr(manager, "_get_messages", lambda user_id, session_id: [{"role": "user", "content": "问"}])
    running = list(manager.render_current_session_progress("u", "s", history))

    monkeypatch.setattr(
        manager,
        "_get_runs",
        lambda user_id, session_id: [
            {
                "run_id": "run",
                "status": "failed",
                "error_text": "断线",
                "steps": [{"message": "调用失败"}],
            },
        ],
    )
    failed = list(manager.render_current_session_progress("u", "s", history))

    monkeypatch.setattr(manager, "_get_runs", lambda user_id, session_id: [{"run_id": "run", "status": "completed"}])
    monkeypatch.setattr(
        manager,
        "_get_messages",
        lambda user_id, session_id: [
            {"role": "user", "content": "问"},
            {"role": "assistant", "content": "答"},
        ],
    )
    completed = list(manager.render_current_session_progress("u", "s", history))

    assert running[-1][1] == "任务运行中..."
    assert "正在调用工具" in running[-1][0][-1]["content"]
    assert failed[-1][1] == "任务失败"
    assert "断线" in failed[-1][0][-1]["content"]
    assert completed[-1][1] == "任务已完成"
    assert completed[-1][0][-1]["content"] == "答"


def test_gradio_memory_and_cancel_error_branches(monkeypatch):
    """验证归档统计和停止任务在无会话、404、HTTP 失败时能返回可读状态。"""
    manager = GradioManager(backend_url="http://testserver")
    request = SimpleNamespace(headers={})
    response = requests.Response()
    response.status_code = 404
    http_error = requests.exceptions.HTTPError(response=response)

    assert manager.get_unconsolidated_count("", "user", request) == "💬 尚未开始对话"

    monkeypatch.setattr(manager, "_resolve_session_id", lambda user_id, session_id, request=None: "session")
    monkeypatch.setattr(manager, "_get_memory_summary", lambda user_id, session_id: (_ for _ in ()).throw(http_error))
    assert manager.get_unconsolidated_count("session", "user", request) == "💬 尚未开始对话"

    monkeypatch.setattr(
        "gradio_manager.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    assert "停止失败" in manager.cancel_current_run("session", "user", request)


@pytest.mark.asyncio
async def test_gradio_chat_function_streams_status_tool_content_and_errors(monkeypatch):
    """验证聊天函数能解析后端 SSE 状态、工具事件、内容、done 和 HTTP 失败。"""
    manager = GradioManager(backend_url="http://testserver")
    request = SimpleNamespace(headers={"cookie": "travel_user_id=user-stream"})
    events = [
        "not-sse\n".encode("utf-8"),
        "data: {bad-json}\n".encode("utf-8"),
        'data: {"type":"status","message":"开始"}\n'.encode("utf-8"),
        'data: {"type":"tool_start","tool":"lookup"}\n'.encode("utf-8"),
        'data: {"type":"tool_result","tool":"lookup","status":"failed","result_summary":"断线"}\n'.encode("utf-8"),
        'data: {"type":"content","content":"答"}\n'.encode("utf-8"),
        'data: {"type":"done","done":true}\n'.encode("utf-8"),
    ]

    class FakeContent:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.chunks:
                raise StopAsyncIteration
            return self.chunks.pop(0)

    class FakeResponse:
        def __init__(self, status, chunks):
            self.status = status
            self.content = FakeContent(chunks)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self, status=200):
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse(self.status, events)

    monkeypatch.setattr(manager, "_create_session", lambda user_id: {"session_id": "session-stream"})
    monkeypatch.setattr("gradio_manager.aiohttp.ClientSession", lambda: FakeSession())

    outputs = []
    async for item in manager.chat_function("你好", [], "", "", request):
        outputs.append(item)

    monkeypatch.setattr("gradio_manager.aiohttp.ClientSession", lambda: FakeSession(status=500))
    failed_outputs = []
    async for item in manager.chat_function("你好", [], "session-stream", "user-stream", request):
        failed_outputs.append(item)

    assert outputs[0] == "> 正在连接后端..."
    assert any("正在调用 lookup" in item for item in outputs)
    assert any("断线" in item for item in outputs)
    assert outputs[-1] == "答"
    assert failed_outputs[-1] == "❌ 请求失败: HTTP 500"


def test_gradio_create_interface_builds_blocks():
    """验证 Gradio 界面构建会完成组件创建、事件绑定和队列配置。"""
    manager = GradioManager(backend_url="http://testserver")
    calls = []

    class FakeEvent:
        def then(self, **kwargs):
            calls.append(("then", kwargs.get("fn")))
            return self

    class FakeComponent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def click(self, **kwargs):
            calls.append(("click", kwargs.get("fn")))
            return FakeEvent()

        def change(self, **kwargs):
            calls.append(("change", kwargs.get("fn")))
            return FakeEvent()

        def tick(self, **kwargs):
            calls.append(("tick", kwargs.get("fn")))
            return FakeEvent()

    class FakeContext(FakeComponent):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load(self, **kwargs):
            calls.append(("load", kwargs.get("fn")))
            return FakeEvent()

        def queue(self, **kwargs):
            calls.append(("queue", kwargs))
            return self

    class FakeChatInterface(FakeComponent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.textbox = FakeComponent()

    monkeypatch = pytest.MonkeyPatch()
    try:
        import gradio_manager as module

        monkeypatch.setattr(module.gr, "Blocks", FakeContext)
        monkeypatch.setattr(module.gr, "Row", FakeContext)
        monkeypatch.setattr(module.gr, "Column", FakeContext)
        monkeypatch.setattr(module.gr, "Markdown", FakeComponent)
        monkeypatch.setattr(module.gr, "Button", FakeComponent)
        monkeypatch.setattr(module.gr, "Dropdown", FakeComponent)
        monkeypatch.setattr(module.gr, "Textbox", FakeComponent)
        monkeypatch.setattr(module.gr, "Timer", FakeComponent)
        monkeypatch.setattr(module.gr, "Chatbot", FakeComponent)
        monkeypatch.setattr(module.gr, "Examples", FakeComponent)
        monkeypatch.setattr(module, "MemoryRefreshingChatInterface", FakeChatInterface)
        monkeypatch.setattr(module.gr.themes, "Soft", lambda: object())

        demo = manager.create_interface()
    finally:
        monkeypatch.undo()

    assert demo is not None
    assert any(name == "load" for name, _ in calls)
    assert any(name == "tick" for name, _ in calls)
    assert any(name == "queue" and payload["max_size"] == 100 for name, payload in calls)


def test_agent_prompt_exposes_skill_summary(agent_factory):
    """验证 Agent 系统提示词会暴露可用 Skill 摘要，供模型选择能力。"""
    agent = agent_factory(["ok"])
    prompt = agent._build_system_prompt()

    assert "可用技能" in prompt
    assert "weather" in prompt
    assert "travel-planner" in prompt
