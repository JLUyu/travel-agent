"""集成测试：通过 ASGITransport 覆盖 FastAPI 后端会话、消息、run、流式和记忆接口。"""

import asyncio
import json

import httpx
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.full]


class FakeBackendAgent:
    """用于替代真实 Agent 的后端流式响应假对象。"""
    def __init__(self, runtime=None, user_id: str = "", session_id: str = ""):
        self.runtime = runtime
        self.user_id = user_id
        self.session_id = session_id
        self.calls = []

    async def chat_stream(self, message, run_id=None, resume=False):
        run_id = run_id or "fake-run"
        self.calls.append({"message": message, "run_id": run_id, "resume": resume})
        if resume and self.runtime:
            self.runtime.record_run_step(run_id, "status", "running", "恢复执行")
        yield {"type": "status", "run_id": run_id, "message": "处理中", "status": "running"}
        for char in "后端回答":
            yield {"type": "content", "run_id": run_id, "content": char}
        if resume and self.runtime:
            self.runtime.record_run_complete(run_id, "completed", "后端回答")
        yield {"type": "done", "run_id": run_id, "done": True, "status": "completed"}

    async def chat_eval(self, message, run_id=None):
        self.calls.append({"message": message, "run_id": run_id, "mode": "eval"})
        return "评测回答", ["lookup"]


class FakeBackendSessionManager:
    """记录后端会话生命周期的假 SessionManager。"""
    def __init__(self, runtime=None):
        self.runtime = runtime
        self.sessions = {}
        self.cleaned = []
        self.created_count = 0

    def create_session(self):
        self.created_count += 1
        if self.created_count == 1:
            return "session-created"
        return f"session-created-{self.created_count}"

    async def get_or_create_agent(self, user_id, session_id, external_urls, local_url):
        self.sessions.setdefault(
            (user_id, session_id),
            FakeBackendAgent(self.runtime, user_id, session_id),
        )
        return self.sessions[(user_id, session_id)]

    def get_agent(self, user_id, session_id):
        return self.sessions[(user_id, session_id)]

    async def cleanup_session(self, user_id, session_id):
        self.cleaned.append((user_id, session_id))
        self.sessions.pop((user_id, session_id), None)


@pytest.fixture
def backend_client(monkeypatch, session_store, fake_runtime):
    """将后端全局依赖替换为 fake 版本，并返回异步测试客户端。"""
    import backend

    manager = FakeBackendSessionManager(fake_runtime)
    monkeypatch.setattr(backend, "session_manager", manager)
    monkeypatch.setattr(backend, "session_store", session_store)
    monkeypatch.setattr(backend, "industrial_runtime", fake_runtime)
    backend.resume_tasks.clear()

    transport = httpx.ASGITransport(app=backend.app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_backend_session_and_health_endpoints(backend_client):
    """验证健康检查、创建会话和会话列表接口的基础响应。"""
    async with backend_client as client:
        health = await client.get("/health")
        created = await client.post("/session/create/user-api")
        listed = await client.get("/session/list/user-api")

    assert health.status_code == 200
    assert created.status_code == 200
    assert listed.json()["current_session_id"] == "session-created"


@pytest.mark.asyncio
async def test_backend_messages_runs_and_stream(backend_client, session_store, fake_runtime):
    """验证历史消息、run 列表和 SSE 风格聊天流输出。"""
    session = session_store.load_session("user-api", "session-api")
    session.add_message("user", "历史问题")
    session.add_message("assistant", "历史回答")
    session_store.save_session(session)
    fake_runtime.record_run_start("run-api", "user-api", "session-api", "输入")
    fake_runtime.record_run_step("run-api", "status", "running", "步骤")

    async with backend_client as client:
        messages = await client.get("/session/messages/user-api/session-api")
        runs = await client.get("/session/runs/user-api/session-api")
        stream = await client.post(
            "/chat/stream",
            json={"user_id": "user-api", "session_id": "session-api", "message": "你好"},
        )

    assert messages.json()["messages"] == [
        {"role": "user", "content": "历史问题"},
        {"role": "assistant", "content": "历史回答"},
    ]
    assert runs.json()["runs"][0]["run_id"] == "run-api"
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in stream.text.splitlines()
        if line.startswith("data: ")
    ]
    assert payloads[0]["type"] == "status"
    assert "".join(p.get("content", "") for p in payloads if p.get("type") == "content") == "后端回答"
    assert payloads[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_backend_chat_eval_uses_lightweight_agent_path(backend_client):
    """验证评测接口走轻量 Agent 路径，并保留与 /chat 兼容的响应结构。"""
    async with backend_client as client:
        response = await client.post(
            "/chat/eval",
            json={"user_id": "eval-user", "session_id": "eval-session", "query": "评测问题"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["finalAnswer"] == "评测回答"
    assert payload["response"] == "评测回答"
    assert payload["tool_calls"] == ["lookup"]


@pytest.mark.asyncio
async def test_backend_cancel_current_run_uses_runtime(backend_client, fake_runtime):
    """验证取消当前 run 会通过 runtime 标记并更新状态。"""
    fake_runtime.record_run_start("run-cancel-api", "user-api", "session-api", "输入")

    async with backend_client as client:
        response = await client.post("/runs/cancel-current/user-api/session-api")

    assert response.status_code == 200
    assert fake_runtime.runs["run-cancel-api"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_backend_memory_summary_reports_counts(backend_client, session_store):
    """验证未归档记忆摘要接口能返回包含统计含义的中文提示。"""
    session = session_store.load_session("user-api", "session-memory")
    session.add_message("user", "a")
    session.add_message("assistant", "b")
    session.update_last_consolidated(1)
    session_store.save_session(session)

    async with backend_client as client:
        response = await client.get("/memory/unconsolidated/user-api/session-memory")

    assert response.status_code == 200
    assert "未归档" in response.json()["summary"] or "鏈綊妗" in response.json()["summary"]


@pytest.mark.asyncio
async def test_backend_first_visit_initializes_session_and_empty_state(backend_client):
    """覆盖首次访问：创建会话、初始化 Agent、读取列表/历史/归档状态。"""
    async with backend_client as client:
        created = await client.post("/session/create/user-first")
        session_id = created.json()["session"]["session_id"]
        initialized = await client.post(f"/session/init/user-first/{session_id}")
        listed = await client.get("/session/list/user-first")
        messages = await client.get(f"/session/messages/user-first/{session_id}")
        memory = await client.get(f"/memory/unconsolidated/user-first/{session_id}")

    assert initialized.json()["status"] == "success"
    assert listed.json()["current_session_id"] == session_id
    assert messages.json()["messages"] == []
    assert "未归档" in memory.json()["summary"]


@pytest.mark.asyncio
async def test_backend_returning_user_restores_last_session_history(
    backend_client,
    session_store,
):
    """覆盖老用户再次访问：会话列表和消息接口能恢复上次会话内容。"""
    session_store.create_session("user-return", "session-return")
    session = session_store.load_session("user-return", "session-return")
    session.add_message("user", "上次的问题")
    session.add_message("assistant", "上次的回答")
    session_store.save_session(session)

    async with backend_client as client:
        listed = await client.get("/session/list/user-return")
        messages = await client.get("/session/messages/user-return/session-return")

    assert listed.json()["current_session_id"] == "session-return"
    assert messages.json()["messages"] == [
        {"role": "user", "content": "上次的问题"},
        {"role": "assistant", "content": "上次的回答"},
    ]


@pytest.mark.asyncio
async def test_backend_switch_session_reads_isolated_histories(
    backend_client,
    session_store,
):
    """覆盖切换会话：同一用户不同 session 的历史互不串读。"""
    for session_id, question, answer in [
        ("session-trip", "上海三日游", "外滩和豫园"),
        ("session-train", "北京到上海", "高铁优先"),
    ]:
        session_store.create_session("user-switch", session_id)
        session = session_store.load_session("user-switch", session_id)
        session.add_message("user", question)
        session.add_message("assistant", answer)
        session_store.save_session(session)

    async with backend_client as client:
        await client.post("/session/init/user-switch/session-trip")
        await client.post("/session/init/user-switch/session-train")
        trip = await client.get("/session/messages/user-switch/session-trip")
        train = await client.get("/session/messages/user-switch/session-train")

    assert trip.json()["messages"][-1]["content"] == "外滩和豫园"
    assert train.json()["messages"][-1]["content"] == "高铁优先"


@pytest.mark.asyncio
async def test_backend_refresh_resumes_running_run_from_checkpoint(
    backend_client,
    fake_runtime,
):
    """覆盖刷新恢复：读取 run 进度时会接管 running run 并从 checkpoint 继续。"""
    import backend

    run_id = "run-resume-api"
    fake_runtime.record_run_start(run_id, "user-resume", "session-resume", "恢复输入")
    fake_runtime.record_run_checkpoint(
        run_id,
        {
            "version": 1,
            "run_id": run_id,
            "next_node": "llm",
            "messages": [{"role": "user", "content": "恢复输入"}],
            "tool_calls": [],
            "final_response": "",
            "retrieval_context": "",
        },
    )

    async with backend_client as client:
        response = await client.get("/session/runs/user-resume/session-resume")
        for _ in range(20):
            if fake_runtime.runs[run_id]["status"] == "completed":
                break
            await asyncio.sleep(0.01)

    assert response.status_code == 200
    assert fake_runtime.runs[run_id]["status"] == "completed"
    agent = backend.session_manager.get_agent("user-resume", "session-resume")
    assert agent.calls[-1]["resume"] is True
    assert agent.calls[-1]["run_id"] == run_id


@pytest.mark.asyncio
async def test_backend_multi_user_streams_keep_sessions_isolated(backend_client):
    """覆盖多用户并发访问：两个用户同时请求时返回各自的 user/session。"""
    async with backend_client as client:
        first, second = await asyncio.gather(
            client.post(
                "/chat/stream",
                json={"user_id": "user-one", "session_id": "session-one", "message": "你好"},
            ),
            client.post(
                "/chat/stream",
                json={"user_id": "user-two", "session_id": "session-two", "message": "你好"},
            ),
        )

    def payloads(response):
        return [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

    assert {item["user_id"] for item in payloads(first)} == {"user-one"}
    assert {item["session_id"] for item in payloads(first)} == {"session-one"}
    assert {item["user_id"] for item in payloads(second)} == {"user-two"}
    assert {item["session_id"] for item in payloads(second)} == {"session-two"}


@pytest.mark.asyncio
async def test_backend_delete_cleanup_legacy_and_noop_paths(backend_client, session_store):
    """覆盖删除会话、清理空会话、legacy 初始化和无任务取消等接口分支。"""
    session_store.create_session("user-clean", "empty-a")
    session_store.create_session("user-clean", "empty-b")
    session_store.create_session("user-clean", "filled")
    filled = session_store.load_session("user-clean", "filled")
    filled.add_message("user", "保留这个会话")
    session_store.save_session(filled)

    async with backend_client as client:
        legacy_init = await client.post("/session/init/legacy-session")
        cleanup = await client.post("/session/cleanup_empty/user-clean?exclude=empty-b")
        deleted_missing = await client.delete("/session/user-clean/not-exists")
        deleted_existing = await client.delete("/session/user-clean/empty-b")
        cancel_none = await client.post("/runs/cancel-current/user-clean/filled")

    assert legacy_init.status_code == 200
    assert "1 个空会话" in cleanup.json()["message"]
    assert deleted_missing.json()["status"] == "noop"
    assert deleted_existing.json()["status"] == "success"
    assert cancel_none.json()["status"] == "noop"
    assert [item["session_id"] for item in session_store.list_sessions("user-clean")] == ["filled"]


@pytest.mark.asyncio
async def test_backend_cancel_current_active_session_run(monkeypatch, backend_client, fake_runtime):
    """覆盖没有 MySQL running run 时，通过 TravelAgent 活跃表取消当前会话任务。"""
    from travel_agent import TravelAgent

    monkeypatch.setattr(
        TravelAgent,
        "cancel_session_run",
        staticmethod(lambda user_id, session_id: ("active-run", False)),
    )

    async with backend_client as client:
        response = await client.post("/runs/cancel-current/user-active/session-active")

    assert response.status_code == 200
    assert response.json()["message"] == "任务已标记为停止"
    assert fake_runtime.runs["active-run"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_backend_memory_summary_agent_initialized_failure_branch(backend_client, session_store):
    """覆盖记忆摘要在 Agent 已初始化但 token 统计失败时的降级展示。"""
    import backend

    session_store.create_session("user-memory-error", "session-memory-error")
    session = session_store.load_session("user-memory-error", "session-memory-error")
    session.add_message("user", "问题")
    session_store.save_session(session)
    await backend.session_manager.get_or_create_agent(
        "user-memory-error",
        "session-memory-error",
        [],
        "",
    )

    async with backend_client as client:
        response = await client.get("/memory/unconsolidated/user-memory-error/session-memory-error")

    assert response.status_code == 200
    assert "统计失败" in response.json()["summary"]


@pytest.mark.asyncio
async def test_backend_start_new_session_success_and_not_found(monkeypatch, backend_client):
    """覆盖兼容新会话接口的成功和缺失 Agent 分支。"""
    import backend

    class StartableAgent(FakeBackendAgent):
        async def start_new_session(self):
            return {"message": "新会话已清空"}

    backend.session_manager.sessions[("user-new", "session-new")] = StartableAgent()
    original_get_agent = backend.session_manager.get_agent

    def fake_get_agent(user_id, session_id):
        if session_id == "missing":
            raise ValueError("会话不存在")
        return original_get_agent(user_id, session_id)

    monkeypatch.setattr(backend.session_manager, "get_agent", fake_get_agent)

    async with backend_client as client:
        success = await client.post("/session/new/user-new/session-new")
        missing = await client.post("/session/new/user-new/missing")

    assert success.json()["message"] == "新会话已清空"
    assert missing.status_code == 404
