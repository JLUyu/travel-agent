"""测试公共配置：提供 fake 运行时、fake LLM/MCP 客户端以及共享 fixture。"""

import asyncio
import configparser
import json
import os
import re
import socket
import threading
import time
import tomllib
import warnings
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("DEEPSEEK_MODEL", "test-model")
os.environ.setdefault("MEMORY_API_KEY", "test-key")
os.environ.setdefault("MEMORY_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("MEMORY_MODEL", "test-memory-model")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")


def _contains_marker_expression(expression: str, marker: str) -> bool:
    """用单词边界匹配 marker，避免把普通字符串片段误判成分层测试。"""
    return re.search(rf"\b{re.escape(marker)}\b", expression or "") is not None


def _format_coverage_value(value):
    """把 pyproject.toml 里的值转换成 coverage rc 文件需要的文本。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        if len(value) == 1:
            return str(value[0])
        return "\n    " + "\n    ".join(str(item) for item in value)
    return str(value)


def _build_layered_coverage_rc_text(pyproject_path: Path, layer: str) -> str:
    """从 pyproject.toml 的自定义分层配置生成 coverage rc 文本。"""
    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)

    coverage_config = (
        pyproject.get("tool", {})
        .get("travel_agent", {})
        .get("coverage", {})
        .get(layer)
    )
    if not coverage_config:
        raise RuntimeError(f"pyproject.toml 缺少 tool.travel_agent.coverage.{layer} 配置")

    parser = configparser.ConfigParser()
    for section_name in ("run", "report"):
        section = coverage_config.get(section_name, {})
        parser[section_name] = {
            key: _format_coverage_value(value)
            for key, value in section.items()
        }

    lines: list[str] = []
    for section_name in parser.sections():
        lines.append(f"[{section_name}]")
        for key, value in parser[section_name].items():
            lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines)


def _write_layered_coverage_config(root_path: Path, layer: str, cache_dir: Path) -> Path:
    """把 pyproject.toml 中的分层配置写成 pytest-cov 可读取的临时 rc 文件。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rc_path = cache_dir / f"coverage-{layer}.rc"
    rc_path.write_text(
        _build_layered_coverage_rc_text(root_path / "pyproject.toml", layer),
        encoding="utf-8",
    )
    return rc_path


def pytest_configure(config):
    """根据 -m unit/integration 自动切换分层 coverage 配置。"""
    option = config.option
    if getattr(option, "cov_config", ".coveragerc") != ".coveragerc":
        return

    expression = getattr(option, "markexpr", "")
    uses_unit = _contains_marker_expression(expression, "unit")
    uses_integration = _contains_marker_expression(expression, "integration")
    if uses_unit and not uses_integration:
        selected_layer = "unit"
    elif uses_integration and not uses_unit:
        selected_layer = "integration"
    else:
        return

    cov_plugin = config.pluginmanager.get_plugin("_cov")
    if cov_plugin is None or getattr(cov_plugin, "_disabled", False):
        return

    # pytest-cov 在 conftest 加载前已按默认配置启动，这里在测试执行前重启到分层口径。
    # 旧控制器此时还没有测试数据，关闭时会产生空数据 warning，属于预期启动切换行为。
    with warnings.catch_warnings():
        try:
            from coverage.exceptions import CoverageWarning

            warnings.filterwarnings("ignore", category=CoverageWarning)
        except Exception:
            pass
        cov_plugin.cov_controller.finish()
    selected_config = _write_layered_coverage_config(
        Path(config.rootpath),
        selected_layer,
        Path(config.rootpath) / ".pytest_cache",
    )
    cov_plugin.options.cov_config = str(selected_config)
    cov_plugin.options.cov_fail_under = None
    cov_plugin.options.cov_precision = None
    cov_plugin.start(type(cov_plugin.cov_controller), config)


@contextmanager
def null_lock():
    """用于 fake runtime 的空锁，模拟真实锁上下文但不做阻塞。"""
    yield


class FakeRuntime:
    """内存版工业运行时替身，用于记录 run、step、checkpoint 和取消状态。"""
    def __init__(self):
        self.runs: dict[str, dict[str, Any]] = {}
        self.steps: list[dict[str, Any]] = []
        self.checkpoints: dict[str, list[dict[str, Any]]] = {}
        self.cancelled: set[str] = set()
        self.enqueued: list[tuple[str, dict[str, Any]]] = []
        self.background_tasks: dict[str, dict[str, Any]] = {}
        self.enqueue_result = False

    def ensure_schema(self):
        return None

    def record_session(self, **kwargs):
        return None

    def record_run_start(self, run_id: str, user_id: str, session_id: str, input_text: str):
        self.runs[run_id] = {
            "run_id": run_id,
            "user_id": user_id,
            "session_id": session_id,
            "status": "running",
            "input_text": input_text,
            "output_text": "",
            "error_text": "",
            "steps": [],
        }

    def record_run_complete(
        self,
        run_id: str,
        status: str,
        output_text: str = "",
        error_text: str = "",
    ):
        self.runs.setdefault(run_id, {"run_id": run_id, "steps": []})
        self.runs[run_id].update(
            {
                "status": status,
                "output_text": output_text,
                "error_text": error_text,
            }
        )

    def record_run_step(
        self,
        run_id: str,
        step_type: str,
        status: str,
        message: str = "",
        tool_name: str | None = None,
        payload: dict[str, Any] | None = None,
        elapsed_ms: int | None = None,
    ):
        step = {
            "run_id": run_id,
            "step_type": step_type,
            "status": status,
            "message": message,
            "tool_name": tool_name,
            "payload_json": payload or {},
            "elapsed_ms": elapsed_ms,
        }
        self.steps.append(step)
        self.runs.setdefault(run_id, {"run_id": run_id, "steps": []})
        self.runs[run_id].setdefault("steps", []).append(step)

    def record_run_checkpoint(self, run_id: str, checkpoint: dict[str, Any]):
        self.checkpoints.setdefault(run_id, []).append(checkpoint)

    def get_latest_run_checkpoint(self, run_id: str):
        checkpoints = self.checkpoints.get(run_id) or []
        return checkpoints[-1] if checkpoints else None

    def get_recent_runs(self, user_id: str, session_id: str, limit: int = 3, **kwargs):
        runs = [
            run
            for run in self.runs.values()
            if run.get("user_id") == user_id and run.get("session_id") == session_id
        ]
        return list(reversed(runs[-limit:]))

    def request_run_cancel(self, run_id: str, reason: str = "用户主动停止任务"):
        self.cancelled.add(run_id)
        self.record_run_step(run_id, "cancelled", "cancelled", reason)
        self.record_run_complete(run_id, "cancelled", error_text=reason)

    def is_run_cancelled(self, run_id: str):
        return run_id in self.cancelled

    def session_lock(self, user_id: str, session_id: str):
        return null_lock()

    def memory_lock(self, user_id: str):
        return null_lock()

    def enqueue_task(self, queue_name: str, payload: dict[str, Any]):
        self.enqueued.append((queue_name, payload))
        return self.enqueue_result

    def record_background_task(
        self,
        task_id: str,
        queue_name: str,
        status: str,
        payload: dict[str, Any] | None = None,
        error_text: str = "",
    ):
        self.background_tasks[task_id] = {
            "task_id": task_id,
            "queue_name": queue_name,
            "status": status,
            "payload": payload or {},
            "error_text": error_text,
        }


class FakeLLMClient:
    """按队列返回预设响应的 LLM 假客户端，避免测试调用真实模型。"""
    def __init__(self, responses):
        self.responses = deque(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    async def _create_completion(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeLLMClient has no queued response")

        item = self.responses.popleft()
        if isinstance(item, Exception):
            raise item

        if isinstance(item, dict) and "tool_calls" in item:
            tool_calls = []
            for call in item["tool_calls"]:
                args = call.get("arguments", {})
                tool_calls.append(
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name=call.get("name", "save_memory"),
                            arguments=json.dumps(args, ensure_ascii=False),
                        )
                    )
                )
            message = SimpleNamespace(content=item.get("content", ""), tool_calls=tool_calls)
        else:
            content = item.get("content", "") if isinstance(item, dict) else str(item)
            message = SimpleNamespace(content=content, tool_calls=[])

        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def close(self):
        return None


class FakeMCPClient:
    """记录工具调用并返回预设结果的 MCP 假客户端。"""
    def __init__(self, results: dict[str, Any] | None = None, description: str = ""):
        self.results = results or {}
        self.description = description or "工具名: lookup\n描述: 测试 MCP 工具"
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tools = {}
        self.tools_version = 1

    def get_tools_description(self):
        return self.description

    async def call_tool(self, tool_name: str, arguments: dict):
        self.calls.append((tool_name, arguments))
        result = self.results.get(tool_name, f"{tool_name} result")
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result(arguments)
        return result

    async def disconnect(self):
        return None


@pytest.fixture
def fake_runtime():
    """为每个测试提供独立的 fake runtime，避免跨用例状态污染。"""
    return FakeRuntime()


@pytest.fixture
def session_store(tmp_path, fake_runtime):
    """创建使用临时目录和 fake runtime 的会话存储。"""
    from session_store import SessionStore

    store = SessionStore(base_path=str(tmp_path / "memory"))
    store.runtime = fake_runtime
    return store


def make_agent(
    tmp_path: Path,
    fake_runtime: FakeRuntime,
    llm_responses,
    mcp_results: dict[str, Any] | None = None,
):
    """构造注入 fake 依赖的 TravelAgent，便于集成流式测试复用。"""
    from session_store import SessionData, SessionStore
    from travel_agent import TravelAgent

    store = SessionStore(base_path=str(tmp_path / "memory"))
    store.runtime = fake_runtime
    user_id = "user-a"
    session_id = "session-a"
    store.create_session(user_id, session_id)

    agent = TravelAgent(
        enable_memory=False,
        user_id=user_id,
        session_id=session_id,
        mcp_client=FakeMCPClient(mcp_results),
    )
    agent.runtime = fake_runtime
    agent.session_store = store
    agent.session = store.load_session(user_id, session_id)
    agent.llm_client = FakeLLMClient(llm_responses)
    if agent.subagent_runner:
        agent.subagent_runner.llm_client = agent.llm_client
        agent.subagent_runner.mcp_client = agent.mcp_client
    return agent


@pytest.fixture
def agent_factory(tmp_path, fake_runtime):
    """返回可按需配置 LLM 响应和 MCP 结果的 Agent 工厂。"""
    def _factory(llm_responses, mcp_results=None):
        return make_agent(tmp_path, fake_runtime, llm_responses, mcp_results)

    return _factory


def free_tcp_port() -> int:
    """向系统申请一个当前可用的本地 TCP 端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_port(port: int, timeout: float = 5.0) -> None:
    """等待本地测试服务端口就绪，超时则显式失败。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError(f"port {port} did not open within {timeout}s")


@pytest.fixture
def local_mcp_url(monkeypatch):
    """启动嵌入式本地 MCP/SSE 服务，并在测试结束后关闭。"""
    import uvicorn
    from fastapi import FastAPI

    import mcp_server

    monkeypatch.setattr(mcp_server, "llm_client", FakeLLMClient(["本地 MCP 摘要"]))

    app = FastAPI()
    mcp_server.register_to_app(app)
    port = free_tcp_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    def run_server():
        asyncio.run(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    wait_for_port(port)
    yield f"http://127.0.0.1:{port}/sse"
    server.should_exit = True
    thread.join(timeout=5)
