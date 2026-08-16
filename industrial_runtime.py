"""可选的工业化集成：持久化、缓存和异步队列。"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

from config import INDUSTRIAL_CONFIG
from logger import info, warn


SENSITIVE_KEYWORDS = ("key", "token", "secret", "password", "passwd", "authorization")


def utc_now_text() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _truncate_json_value(value: Any, limit: int) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _truncate_json_value(item, limit)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_truncate_json_value(item, limit) for item in value[:50]]
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...(truncated)"
    return value


def safe_json(value: Any, limit: int = 800) -> str:
    try:
        text = json.dumps(_truncate_json_value(value, limit), ensure_ascii=False, default=str)
    except Exception:
        text = json.dumps(str(value), ensure_ascii=False)
    if len(text) > limit:
        return json.dumps(
            {
                "_truncated": True,
                "original_length": len(text),
                "preview": text[:limit],
            },
            ensure_ascii=False,
        )
    return text


def full_json(value: Any) -> str:
    """序列化不可截断的持久化载荷。"""
    return json.dumps(value, ensure_ascii=False, default=str)


def sanitize_payload(value: Any, limit: int = 300) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if any(keyword in str(key).lower() for keyword in SENSITIVE_KEYWORDS):
                cleaned[key] = "***"
            else:
                cleaned[key] = sanitize_payload(item, limit=limit)
        return cleaned
    if isinstance(value, list):
        return [sanitize_payload(item, limit=limit) for item in value[:20]]
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...(truncated)"
    return value


class IndustrialRuntime:
    """MySQL、Redis 和 RabbitMQ 的尽力适配器。

    所有集成均为可选。当依赖或外部服务不可用时，
    方法降级为空操作，保持本地开发流程正常运作。
    """

    def __init__(self):
        self.config = INDUSTRIAL_CONFIG
        self._schema_ready = False
        self._mysql_disabled_reason: str | None = None
        self._redis_client = None
        self._redis_disabled_reason: str | None = None
        self._rabbit_disabled_reason: str | None = None

    @property
    def mysql_enabled(self) -> bool:
        return bool(self.config["mysql"].get("enabled")) and not self._mysql_disabled_reason

    @property
    def redis_enabled(self) -> bool:
        return bool(self.config["redis"].get("enabled")) and not self._redis_disabled_reason

    @property
    def rabbitmq_enabled(self) -> bool:
        return bool(self.config["rabbitmq"].get("enabled")) and not self._rabbit_disabled_reason

    def _mysql_connect(self):
        if not self.config["mysql"].get("enabled"):
            return None
        if self._mysql_disabled_reason:
            return None
        try:
            import pymysql
        except Exception as exc:
            self._mysql_disabled_reason = f"pymysql unavailable: {exc}"
            warn("Infra", f"MySQL disabled: {self._mysql_disabled_reason}")
            return None

        cfg = self.config["mysql"]
        try:
            return pymysql.connect(
                host=cfg["host"],
                port=cfg["port"],
                user=cfg["user"],
                password=cfg["password"],
                database=cfg["database"],
                charset=cfg.get("charset", "utf8mb4"),
                autocommit=True,
                connect_timeout=3,
                read_timeout=5,
                write_timeout=5,
            )
        except Exception as exc:
            self._mysql_disabled_reason = str(exc)
            warn("Infra", f"MySQL disabled: {exc}")
            return None

    def ensure_schema(self) -> None:
        if self._schema_ready or not self.config["mysql"].get("enabled"):
            return
        conn = self._mysql_connect()
        if conn is None:
            return
        statements = [
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(128) PRIMARY KEY,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(128) PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                title VARCHAR(255) NOT NULL,
                status VARCHAR(32) NOT NULL DEFAULT 'active',
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                INDEX idx_sessions_user_updated (user_id, updated_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id VARCHAR(64) PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                session_id VARCHAR(128) NOT NULL,
                status VARCHAR(32) NOT NULL,
                input_text MEDIUMTEXT NULL,
                output_text MEDIUMTEXT NULL,
                error_text TEXT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                INDEX idx_runs_session_created (session_id, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_run_steps (
                step_id VARCHAR(64) PRIMARY KEY,
                run_id VARCHAR(64) NOT NULL,
                step_type VARCHAR(64) NOT NULL,
                status VARCHAR(32) NOT NULL,
                message TEXT NULL,
                tool_name VARCHAR(128) NULL,
                payload_json JSON NULL,
                elapsed_ms INT NULL,
                created_at DATETIME NOT NULL,
                INDEX idx_steps_run_created (run_id, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS background_tasks (
                task_id VARCHAR(64) PRIMARY KEY,
                queue_name VARCHAR(128) NOT NULL,
                status VARCHAR(32) NOT NULL,
                payload_json JSON NULL,
                error_text TEXT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                INDEX idx_tasks_queue_status (queue_name, status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        ]
        try:
            with conn.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            self._schema_ready = True
            info("Infra", "MySQL schema ready")
        except Exception as exc:
            self._mysql_disabled_reason = str(exc)
            warn("Infra", f"MySQL schema init failed: {exc}")
        finally:
            conn.close()

    def _execute_mysql(self, sql: str, params: tuple[Any, ...]) -> None:
        if not self.config["mysql"].get("enabled"):
            return
        self.ensure_schema()
        if not self.mysql_enabled:
            return
        conn = self._mysql_connect()
        if conn is None:
            return
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
        except Exception as exc:
            warn("Infra", f"MySQL write failed: {exc}")
        finally:
            conn.close()

    def _query_mysql(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if not self.config["mysql"].get("enabled"):
            return []
        self.ensure_schema()
        if not self.mysql_enabled:
            return []
        conn = self._mysql_connect()
        if conn is None:
            return []
        try:
            import pymysql

            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql, params)
                return list(cursor.fetchall())
        except Exception as exc:
            warn("Infra", f"MySQL query failed: {exc}")
            return []
        finally:
            conn.close()

    def record_session(
        self,
        user_id: str,
        session_id: str,
        title: str,
        created_at: str | None,
        updated_at: str | None,
    ) -> None:
        created = self._mysql_time(created_at)
        updated = self._mysql_time(updated_at)
        self._execute_mysql(
            """
            INSERT INTO users (user_id, created_at, updated_at)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE updated_at = VALUES(updated_at)
            """,
            (user_id, created, updated),
        )
        self._execute_mysql(
            """
            INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                status = 'active',
                updated_at = VALUES(updated_at)
            """,
            (session_id, user_id, title, created, updated),
        )

    def record_run_start(
        self,
        run_id: str,
        user_id: str,
        session_id: str,
        input_text: str,
    ) -> None:
        now = utc_now_text()
        self._execute_mysql(
            """
            INSERT INTO agent_runs
                (run_id, user_id, session_id, status, input_text, created_at, updated_at)
            VALUES (%s, %s, %s, 'running', %s, %s, %s)
            ON DUPLICATE KEY UPDATE status = 'running', updated_at = VALUES(updated_at)
            """,
            (run_id, user_id, session_id, input_text, now, now),
        )

    def record_run_complete(
        self,
        run_id: str,
        status: str,
        output_text: str = "",
        error_text: str = "",
    ) -> None:
        self._execute_mysql(
            """
            UPDATE agent_runs
            SET status = %s, output_text = %s, error_text = %s, updated_at = %s
            WHERE run_id = %s
            """,
            (status, output_text, error_text, utc_now_text(), run_id),
        )

    def request_run_cancel(self, run_id: str, reason: str = "用户主动停止任务") -> None:
        """将运行标记为已取消（持久化和实时状态）。"""
        self.record_run_step(
            run_id=run_id,
            step_type="cancelled",
            status="cancelled",
            message=reason,
        )
        self.record_run_complete(run_id, "cancelled", error_text=reason)
        client = self._redis()
        if client is None:
            return
        try:
            ttl = int(self.config["redis"].get("ttl_seconds", 3600))
            client.setex(f"run:{run_id}:cancelled", ttl, "1")
        except Exception as exc:
            warn("Infra", f"Redis cancel write failed: {exc}")

    def is_run_cancelled(self, run_id: str) -> bool:
        client = self._redis()
        if client is None:
            return False
        try:
            return bool(client.get(f"run:{run_id}:cancelled"))
        except Exception as exc:
            warn("Infra", f"Redis cancel read failed: {exc}")
            return False

    def record_run_step(
        self,
        run_id: str,
        step_type: str,
        status: str,
        message: str = "",
        tool_name: str | None = None,
        payload: dict[str, Any] | None = None,
        elapsed_ms: int | None = None,
    ) -> None:
        self._execute_mysql(
            """
            INSERT INTO agent_run_steps
                (step_id, run_id, step_type, status, message, tool_name, payload_json, elapsed_ms, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                uuid.uuid4().hex,
                run_id,
                step_type,
                status,
                message,
                tool_name,
                safe_json(payload or {}),
                elapsed_ms,
                utc_now_text(),
            ),
        )

    def record_run_checkpoint(self, run_id: str, checkpoint: dict[str, Any]) -> None:
        """持久化运行最新的可恢复 Agent 状态。

        恢复点是内部恢复记录，因此不会更新用户可见的进度消息。
        """
        if not run_id:
            return
        try:
            payload_json = full_json(checkpoint or {})
        except Exception as exc:
            warn("Infra", f"Run checkpoint serialize failed: {exc}")
            return

        self._execute_mysql(
            """
            INSERT INTO agent_run_steps
                (step_id, run_id, step_type, status, message, tool_name, payload_json, elapsed_ms, created_at)
            VALUES (%s, %s, 'checkpoint', 'saved', '', NULL, %s, NULL, %s)
            """,
            (
                uuid.uuid4().hex,
                run_id,
                payload_json,
                utc_now_text(),
            ),
        )

    def get_latest_run_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """返回运行最近的持久化 Agent 恢复点。"""
        if not run_id:
            return None
        rows = self._query_mysql(
            """
            SELECT payload_json
            FROM agent_run_steps
            WHERE run_id = %s
              AND step_type = 'checkpoint'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (run_id,),
        )
        if not rows:
            return None

        payload = rows[0].get("payload_json")
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
                return data if isinstance(data, dict) else None
            except Exception as exc:
                warn("Infra", f"Run checkpoint parse failed: {exc}")
                return None
        return None

    def get_recent_runs(
        self,
        user_id: str,
        session_id: str,
        limit: int = 3,
        include_recent_minutes: int = 30,
    ) -> list[dict[str, Any]]:
        cutoff = (datetime.utcnow() - timedelta(minutes=include_recent_minutes)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        runs = self._query_mysql(
            """
            SELECT run_id, user_id, session_id, status, input_text, output_text,
                   error_text, created_at, updated_at
            FROM agent_runs
            WHERE user_id = %s
              AND session_id = %s
              AND (status = 'running' OR updated_at >= %s)
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, session_id, cutoff, limit),
        )
        for run in runs:
            steps = self._query_mysql(
                """
                SELECT step_type, status, message, tool_name, payload_json,
                       elapsed_ms, created_at
                FROM agent_run_steps
                WHERE run_id = %s
                ORDER BY created_at ASC
                """,
                (run["run_id"],),
            )
            for step in steps:
                payload = step.get("payload_json")
                if isinstance(payload, str):
                    try:
                        step["payload_json"] = json.loads(payload)
                    except Exception:
                        step["payload_json"] = {}
            run["steps"] = steps
        return runs

    def _redis(self):
        if not self.config["redis"].get("enabled"):
            return None
        if self._redis_disabled_reason:
            return None
        if self._redis_client is not None:
            return self._redis_client
        try:
            import redis
        except Exception as exc:
            self._redis_disabled_reason = f"redis unavailable: {exc}"
            warn("Infra", f"Redis disabled: {self._redis_disabled_reason}")
            return None
        try:
            self._redis_client = redis.Redis.from_url(
                self.config["redis"]["url"],
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._redis_client.ping()
            info("Infra", "Redis connected")
            return self._redis_client
        except Exception as exc:
            self._redis_disabled_reason = str(exc)
            warn("Infra", f"Redis disabled: {exc}")
            return None

    @contextmanager
    def session_lock(self, user_id: str, session_id: str):
        client = self._redis()
        lock = None
        if client is not None:
            try:
                lock = client.lock(
                    f"lock:session:{user_id}:{session_id}",
                    timeout=180,
                    blocking_timeout=5,
                )
                lock.acquire()
            except Exception as exc:
                warn("Infra", f"Redis lock unavailable: {exc}")
                lock = None
        try:
            yield
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    @contextmanager
    def memory_lock(self, user_id: str):
        client = self._redis()
        lock = None
        if client is not None:
            try:
                lock = client.lock(
                    f"lock:memory:{user_id}",
                    timeout=900,
                    blocking_timeout=30,
                )
                lock.acquire()
            except Exception as exc:
                warn("Infra", f"Redis memory lock unavailable: {exc}")
                lock = None
        try:
            yield
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def enqueue_task(self, queue_name: str, payload: dict[str, Any]) -> bool:
        task_id = payload.get("task_id") or uuid.uuid4().hex
        payload = {**payload, "task_id": task_id}
        if not self.config["rabbitmq"].get("enabled") or self._rabbit_disabled_reason:
            return False
        try:
            import pika
        except Exception as exc:
            self._rabbit_disabled_reason = f"pika unavailable: {exc}"
            warn("Infra", f"RabbitMQ disabled: {self._rabbit_disabled_reason}")
            return False

        try:
            params = pika.URLParameters(self.config["rabbitmq"]["url"])
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=queue_name, durable=True)
            channel.basic_publish(
                exchange="",
                routing_key=queue_name,
                body=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2),
            )
            connection.close()
            self.record_background_task(task_id, queue_name, "queued", payload)
            return True
        except Exception as exc:
            self._rabbit_disabled_reason = str(exc)
            warn("Infra", f"RabbitMQ enqueue failed: {exc}")
            return False

    def record_background_task(
        self,
        task_id: str,
        queue_name: str,
        status: str,
        payload: dict[str, Any] | None = None,
        error_text: str = "",
    ) -> None:
        now = utc_now_text()
        self._execute_mysql(
            """
            INSERT INTO background_tasks
                (task_id, queue_name, status, payload_json, error_text, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                status = VALUES(status),
                payload_json = VALUES(payload_json),
                error_text = VALUES(error_text),
                updated_at = VALUES(updated_at)
            """,
            (task_id, queue_name, status, safe_json(payload or {}), error_text, now, now),
        )

    def _mysql_time(self, value: str | None) -> str:
        if not value:
            return utc_now_text()
        try:
            return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return utc_now_text()


_runtime: IndustrialRuntime | None = None


def get_industrial_runtime() -> IndustrialRuntime:
    global _runtime
    if _runtime is None:
        _runtime = IndustrialRuntime()
    return _runtime
