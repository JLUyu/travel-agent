"""RabbitMQ Worker 入口，用于工业化后台任务处理。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from config import INDUSTRIAL_CONFIG
from industrial_runtime import get_industrial_runtime
from logger import error, info, ok, warn
from memory_consolidator import MemoryConsolidator
from session_store import SessionStore


class IndustrialWorker:
    def __init__(self):
        self.runtime = get_industrial_runtime()
        self.rabbitmq_url = INDUSTRIAL_CONFIG["rabbitmq"]["url"]

    async def handle_memory_consolidate(self, payload: dict[str, Any]) -> None:
        task_id = payload.get("task_id", "")
        user_id = payload.get("user_id")
        session_id = payload.get("session_id")
        run_id = payload.get("run_id")

        if not user_id or not session_id:
            raise ValueError("memory.consolidate payload requires user_id and session_id")

        self.runtime.record_background_task(
            task_id,
            "memory.consolidate",
            "running",
            payload,
        )
        if run_id and self.runtime.is_run_cancelled(str(run_id)):
            self.runtime.record_background_task(
                task_id,
                "memory.consolidate",
                "cancelled",
                payload,
            )
            info("Worker", f"任务已跳过，来源 run 已停止 | task={task_id}")
            return

        session_store = SessionStore()
        session = session_store.load_session(user_id, session_id)
        consolidator = MemoryConsolidator(
            session_store=session_store,
            vector_store=None,
            user_id=user_id,
            context_window_tokens=int(payload.get("context_window_tokens") or 9000),
            max_completion_tokens=int(payload.get("max_completion_tokens") or 1024),
            safety_buffer=int(payload.get("safety_buffer") or 1024),
        )
        success = False
        try:
            await consolidator.maybe_consolidate(
                session,
                payload.get("system_prompt") or "",
                request_id=run_id,
            )
            success = True
        except Exception as exc:
            if run_id:
                self.runtime.record_run_step(
                    run_id=str(run_id),
                    step_type="memory_consolidate",
                    status="failed",
                    message=f"长期记忆归档失败: {exc}",
                    payload={"task_id": task_id},
                )
            raise
        finally:
            await consolidator.close()

        if run_id:
            self.runtime.record_run_step(
                run_id=str(run_id),
                step_type="memory_consolidate",
                status="completed",
                message="长期记忆归档已完成" if success else "长期记忆归档未产生更新",
                payload={"task_id": task_id},
            )

        self.runtime.record_background_task(
            task_id,
            "memory.consolidate",
            "completed",
            payload,
        )

    async def handle_message(self, queue_name: str, payload: dict[str, Any]) -> None:
        if queue_name == "memory.consolidate":
            await self.handle_memory_consolidate(payload)
            return
        warn("Worker", f"未知后台任务队列: {queue_name}")

    def run(self) -> None:
        if not INDUSTRIAL_CONFIG["rabbitmq"]["enabled"]:
            raise RuntimeError("RABBITMQ_ENABLED=false，后台 Worker 未启动")

        try:
            import pika
        except Exception as exc:
            raise RuntimeError(f"缺少 RabbitMQ 依赖 pika: {exc}") from exc

        params = pika.URLParameters(self.rabbitmq_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue="memory.consolidate", durable=True)
        channel.basic_qos(prefetch_count=1)

        def callback(ch, method, _properties, body):
            queue_name = method.routing_key
            payload = {}
            try:
                payload = json.loads(body.decode("utf-8"))
                asyncio.run(self.handle_message(queue_name, payload))
                ch.basic_ack(delivery_tag=method.delivery_tag)
                ok("Worker", f"任务完成 | queue={queue_name} task={payload.get('task_id', '')}")
            except Exception as exc:
                task_id = payload.get("task_id", "") if isinstance(payload, dict) else ""
                self.runtime.record_background_task(
                    task_id,
                    queue_name,
                    "failed",
                    payload if isinstance(payload, dict) else {},
                    str(exc),
                )
                error("Worker", f"任务失败 | queue={queue_name} err={exc}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        channel.basic_consume(queue="memory.consolidate", on_message_callback=callback)
        info("Worker", "RabbitMQ Worker 已启动，监听 memory.consolidate")
        channel.start_consuming()


if __name__ == "__main__":
    IndustrialWorker().run()
