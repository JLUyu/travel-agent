"""全量测试：连接真实 MySQL、Redis、RabbitMQ 验证工业运行时能力。"""

import json
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from industrial_runtime import IndustrialRuntime
from session_store import SessionData, SessionStore


pytestmark = [pytest.mark.integration, pytest.mark.full]


def _runtime_for_infra() -> IndustrialRuntime:
    """创建真实 IndustrialRuntime，供外部中间件测试复用。"""
    runtime = IndustrialRuntime()
    return runtime


def _mysql_runtime_or_skip() -> IndustrialRuntime:
    """按当前配置自动探测 MySQL，可连接才运行真实 MySQL 测试。"""
    runtime = _runtime_for_infra()
    runtime.config["mysql"]["enabled"] = True
    runtime.ensure_schema()
    if not runtime.mysql_enabled:
        pytest.skip("MySQL is not reachable with current test configuration")
    return runtime


def _redis_runtime_or_skip() -> tuple[IndustrialRuntime, object]:
    """按当前配置自动探测 Redis，可连接才运行真实 Redis 测试。"""
    runtime = _runtime_for_infra()
    runtime.config["redis"]["enabled"] = True
    client = runtime._redis()
    if client is None:
        pytest.skip("Redis is not reachable with current test configuration")
    return runtime, client


def _rabbitmq_runtime_or_skip() -> tuple[IndustrialRuntime, object]:
    """按当前配置自动探测 RabbitMQ，可连接才运行真实 RabbitMQ 测试。"""
    runtime = _runtime_for_infra()
    runtime.config["rabbitmq"]["enabled"] = True
    try:
        import pika
    except Exception as exc:
        pytest.skip(f"pika unavailable: {exc}")

    try:
        params = pika.URLParameters(runtime.config["rabbitmq"]["url"])
        connection = pika.BlockingConnection(params)
        connection.close()
    except Exception as exc:
        pytest.skip(f"RabbitMQ is not reachable with current test configuration: {exc}")
    return runtime, pika


def test_mysql_records_sessions_runs_steps_and_checkpoints():
    """验证 MySQL 持久化 session、run、step、checkpoint 并可读取最近 run。"""
    runtime = _mysql_runtime_or_skip()

    suffix = uuid.uuid4().hex[:12]
    user_id = f"test-user-{suffix}"
    session_id = f"test-session-{suffix}"
    run_id = f"test-run-{suffix}"
    checkpoint = {
        "version": 1,
        "run_id": run_id,
        "next_node": "llm",
        "messages": [{"role": "user", "content": "x" * 2000}],
        "tool_calls": [],
        "final_response": "",
        "retrieval_context": "",
    }

    runtime.record_session(user_id, session_id, "测试会话", None, None)
    runtime.record_run_start(run_id, user_id, session_id, "测试输入")
    runtime.record_run_step(run_id, "status", "running", "开始")
    runtime.record_run_checkpoint(run_id, checkpoint)
    runtime.record_run_complete(run_id, "completed", "测试输出")

    latest = runtime.get_latest_run_checkpoint(run_id)
    runs = runtime.get_recent_runs(user_id, session_id, limit=1)

    assert latest == checkpoint
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "completed"
    assert any(step["step_type"] == "checkpoint" for step in runs[0]["steps"])


def test_mysql_running_run_can_be_completed_after_checkpoint_resume():
    """验证真实 MySQL 中 running run 可读取 checkpoint，并在接管后更新为 completed。"""
    runtime = _mysql_runtime_or_skip()

    suffix = uuid.uuid4().hex[:12]
    user_id = f"test-resume-user-{suffix}"
    session_id = f"test-resume-session-{suffix}"
    run_id = f"test-resume-run-{suffix}"
    checkpoint = {
        "version": 1,
        "run_id": run_id,
        "next_node": "llm",
        "messages": [{"role": "user", "content": "恢复输入"}],
        "tool_calls": [],
        "final_response": "",
        "retrieval_context": "",
    }

    runtime.record_session(user_id, session_id, "恢复测试", None, None)
    runtime.record_run_start(run_id, user_id, session_id, "恢复输入")
    runtime.record_run_checkpoint(run_id, checkpoint)

    running = runtime.get_recent_runs(user_id, session_id, limit=1)[0]
    latest = runtime.get_latest_run_checkpoint(run_id)
    runtime.record_run_step(run_id, "status", "running", "服务重启后正在接管任务")
    runtime.record_run_complete(run_id, "completed", "恢复后的回答")
    completed = runtime.get_recent_runs(user_id, session_id, limit=1)[0]

    assert running["status"] == "running"
    assert latest == checkpoint
    assert completed["status"] == "completed"
    assert completed["output_text"] == "恢复后的回答"


def test_redis_cancel_flag_ttl_and_locks():
    """验证 Redis 取消标记 TTL，以及会话锁/记忆锁的互斥行为。"""
    runtime, client = _redis_runtime_or_skip()

    run_id = f"test-redis-{uuid.uuid4().hex[:12]}"
    runtime.request_run_cancel(run_id, reason="测试取消")

    assert runtime.is_run_cancelled(run_id)
    assert client.ttl(f"run:{run_id}:cancelled") > 0

    order = []

    def locked_worker(index: int):
        with runtime.session_lock("test-user", "test-session"):
            order.append(f"start-{index}")
            time.sleep(0.05)
            order.append(f"end-{index}")

    first = threading.Thread(target=locked_worker, args=(1,))
    second = threading.Thread(target=locked_worker, args=(2,))
    first.start()
    second.start()
    first.join()
    second.join()

    assert order in (
        ["start-1", "end-1", "start-2", "end-2"],
        ["start-2", "end-2", "start-1", "end-1"],
    )
    with runtime.memory_lock("test-user"):
        assert True


def test_rabbitmq_enqueue_and_consume_memory_task():
    """验证 RabbitMQ 记忆归档任务能入队、消费、ack 并清理测试队列。"""
    runtime, pika = _rabbitmq_runtime_or_skip()

    queue_name = f"memory.consolidate.test.{uuid.uuid4().hex[:12]}"
    payload = {
        "run_id": f"run-{uuid.uuid4().hex[:8]}",
        "user_id": "test-user",
        "session_id": "test-session",
    }

    assert runtime.enqueue_task(queue_name, payload)

    params = pika.URLParameters(runtime.config["rabbitmq"]["url"])
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    method, properties, body = channel.basic_get(queue=queue_name, auto_ack=False)
    try:
        assert method is not None
        decoded = json.loads(body.decode("utf-8"))
        assert decoded["run_id"] == payload["run_id"]
        assert decoded["task_id"]
        channel.basic_ack(method.delivery_tag)
    finally:
        channel.queue_delete(queue=queue_name)
        connection.close()


@pytest.mark.asyncio
async def test_rabbitmq_memory_worker_records_completion_step(monkeypatch, tmp_path):
    """验证 RabbitMQ 消费出的归档任务经 worker 处理后会写 memory_consolidate step。"""
    runtime = _mysql_runtime_or_skip()
    runtime.config["rabbitmq"]["enabled"] = True
    _rabbitmq_runtime, pika = _rabbitmq_runtime_or_skip()

    import industrial_worker

    suffix = uuid.uuid4().hex[:12]
    user_id = f"test-worker-user-{suffix}"
    session_id = f"test-worker-session-{suffix}"
    run_id = f"test-worker-run-{suffix}"
    queue_name = f"memory.consolidate.worker.test.{suffix}"

    store = SessionStore(base_path=str(tmp_path / "memory"))
    store.runtime = runtime
    session = SessionData(session_id=session_id, user_id=user_id)
    session.add_message("user", "我喜欢高铁和博物馆。" * 20)
    session.add_message("assistant", "已记录偏好。" * 20)
    session.add_message("user", "继续规划后续行程。" * 20)
    store.save_session(session)

    class FakeWorkerConsolidator:
        def __init__(self, session_store, vector_store, user_id, **kwargs):
            self.session_store = session_store
            self.user_id = user_id

        async def maybe_consolidate(self, session, system_prompt="", request_id=None):
            self.session_store.write_memory_md(self.user_id, "用户偏好：高铁、博物馆")
            self.session_store.append_history_md(self.user_id, "[2026-01-01 10:00] 用户讨论高铁和博物馆")
            session.update_last_consolidated(2)
            self.session_store.save_session(session)
            return True

        async def close(self):
            return None

    monkeypatch.setattr(industrial_worker, "SessionStore", lambda: store)
    monkeypatch.setattr(industrial_worker, "MemoryConsolidator", FakeWorkerConsolidator)
    monkeypatch.setattr(industrial_worker, "get_industrial_runtime", lambda: runtime)

    runtime.record_session(user_id, session_id, "worker 测试", None, None)
    runtime.record_run_start(run_id, user_id, session_id, "触发归档")
    payload = {
        "run_id": run_id,
        "user_id": user_id,
        "session_id": session_id,
    }

    assert runtime.enqueue_task(queue_name, payload)
    params = pika.URLParameters(runtime.config["rabbitmq"]["url"])
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    method, _properties, body = channel.basic_get(queue=queue_name, auto_ack=False)
    try:
        assert method is not None
        decoded = json.loads(body.decode("utf-8"))
        channel.basic_ack(method.delivery_tag)
    finally:
        channel.queue_delete(queue=queue_name)
        connection.close()

    worker = industrial_worker.IndustrialWorker()
    await worker.handle_memory_consolidate(decoded)
    runs = runtime.get_recent_runs(user_id, session_id, limit=1)
    loaded = store.load_session(user_id, session_id)

    assert loaded.last_consolidated == 2
    assert "高铁" in store.read_memory_md(user_id)
    assert any(
        step["step_type"] == "memory_consolidate" and step["status"] == "completed"
        for step in runs[0]["steps"]
    )


@pytest.mark.asyncio
async def test_worker_handles_unknown_missing_cancelled_and_failed_tasks(monkeypatch):
    """验证后台 Worker 的未知队列、缺字段、取消和失败记录分支。"""
    import industrial_worker

    class WorkerRuntime:
        def __init__(self):
            self.cancelled = {"run-cancel"}
            self.background = []
            self.steps = []

        def record_background_task(self, task_id, queue_name, status, payload=None, error_text=""):
            self.background.append((task_id, queue_name, status, payload or {}, error_text))

        def is_run_cancelled(self, run_id):
            return run_id in self.cancelled

        def record_run_step(self, **kwargs):
            self.steps.append(kwargs)

    class FailingConsolidator:
        def __init__(self, *args, **kwargs):
            pass

        async def maybe_consolidate(self, *args, **kwargs):
            raise RuntimeError("consolidate failed")

        async def close(self):
            return None

    class FakeWorkerStore:
        def load_session(self, user_id, session_id):
            return SimpleNamespace(session_id=session_id, messages=[])

    runtime = WorkerRuntime()
    monkeypatch.setattr(industrial_worker, "get_industrial_runtime", lambda: runtime)
    monkeypatch.setattr(industrial_worker, "MemoryConsolidator", FailingConsolidator)
    monkeypatch.setattr(industrial_worker, "SessionStore", lambda: FakeWorkerStore())

    worker = industrial_worker.IndustrialWorker()
    await worker.handle_message("unknown.queue", {"task_id": "unknown"})

    with pytest.raises(ValueError):
        await worker.handle_memory_consolidate({"task_id": "missing"})

    await worker.handle_memory_consolidate(
        {
            "task_id": "cancelled",
            "user_id": "u",
            "session_id": "s",
            "run_id": "run-cancel",
        }
    )

    with pytest.raises(RuntimeError, match="consolidate failed"):
        await worker.handle_memory_consolidate(
            {
                "task_id": "failed",
                "user_id": "u",
                "session_id": "s",
                "run_id": "run-fail",
            }
        )

    assert ("cancelled", "memory.consolidate", "cancelled", {"task_id": "cancelled", "user_id": "u", "session_id": "s", "run_id": "run-cancel"}, "") in runtime.background
    assert any(step["status"] == "failed" and step["step_type"] == "memory_consolidate" for step in runtime.steps)
