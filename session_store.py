"""
会话存储模块 - 管理用户级记忆和会话级 JSONL 文件
"""
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from industrial_runtime import get_industrial_runtime
from logger import warn


class SessionData:
    """会话数据容器"""

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.user_id = user_id or session_id
        self.session_id = session_id
        self.messages: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.last_consolidated = 0  # 已整合的消息数量

    def add_message(self, role: str, content: str, **kwargs):
        """添加消息"""
        msg = {
            "message_id": kwargs.pop("message_id", str(uuid.uuid4())),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()
        return msg

    def get_unconsolidated_messages(self) -> List[Dict[str, Any]]:
        """获取未整合的消息（last_consolidated 之后的消息）"""
        return self.messages[self.last_consolidated :]

    def update_last_consolidated(self, count: int):
        """更新 last_consolidated 指针"""
        self.last_consolidated = count
        self.updated_at = datetime.now()


class SessionStore:
    """会话存储管理器 - 用户级 MEMORY/HISTORY，共享多个会话 JSONL。"""

    def __init__(self, base_path: str = "./memory"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.runtime = get_industrial_runtime()
        self.runtime.ensure_schema()

    def _get_user_dir(self, user_id: str) -> Path:
        user_dir = self.base_path / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _get_visible_memory_dir(self, user_id: str) -> Path:
        visible_dir = self._get_user_dir(user_id) / "visible_memory"
        visible_dir.mkdir(parents=True, exist_ok=True)
        return visible_dir

    def _get_sessions_dir(self, user_id: str) -> Path:
        sessions_dir = self._get_user_dir(user_id) / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir

    def _get_session_dir(self, user_id: str, session_id: str) -> Path:
        session_dir = self._get_sessions_dir(user_id) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _get_jsonl_path(self, user_id: str, session_id: str) -> Path:
        """获取会话 JSONL 文件路径"""
        return self._get_session_dir(user_id, session_id) / "session.jsonl"

    def _get_legacy_jsonl_path(self, user_id: str) -> Path:
        """旧版 user_id=session_id 时的会话路径。"""
        return self._get_visible_memory_dir(user_id) / "session.jsonl"

    def _get_index_path(self, user_id: str) -> Path:
        return self._get_user_dir(user_id) / "sessions.json"

    def _read_index(self, user_id: str) -> list[dict[str, Any]]:
        index_path = self._get_index_path(user_id)
        if not index_path.exists():
            return []

        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except Exception as e:
            warn("Store", f"读取用户 {user_id} 会话索引失败: {e}")

        return []

    def _write_index(self, user_id: str, sessions: list[dict[str, Any]]):
        index_path = self._get_index_path(user_id)
        try:
            index_path.write_text(
                json.dumps(sessions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            warn("Store", f"写入用户 {user_id} 会话索引失败: {e}")

    def _default_title(self, session_id: str, created_at: str | None = None) -> str:
        suffix = session_id[:8]
        if created_at:
            try:
                created = datetime.fromisoformat(created_at)
                return f"{created.strftime('%m-%d %H:%M')} 会话 {suffix}"
            except ValueError:
                pass
        return f"会话 {suffix}"

    def _title_from_messages(self, session: SessionData) -> str | None:
        for msg in session.messages:
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", "")).strip()
            if content:
                return content[:24]
        return None

    def _upsert_session_index(
        self,
        user_id: str,
        session_id: str,
        *,
        title: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        sessions = self._read_index(user_id)
        now = datetime.now().isoformat()
        created_at = created_at or now
        updated_at = updated_at or now
        found = False

        for item in sessions:
            if item.get("session_id") != session_id:
                continue
            item["created_at"] = item.get("created_at") or created_at
            item["updated_at"] = updated_at
            if title:
                item["title"] = title
            elif not item.get("title"):
                item["title"] = self._default_title(session_id, item.get("created_at"))
            found = True
            break

        if not found:
            sessions.append(
                {
                    "session_id": session_id,
                    "title": title or self._default_title(session_id, created_at),
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )

        sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        self._write_index(user_id, sessions)
        current = next(
            (item for item in sessions if item.get("session_id") == session_id),
            None,
        )
        if current:
            self.runtime.record_session(
                user_id=user_id,
                session_id=session_id,
                title=str(current.get("title") or self._default_title(session_id, created_at)),
                created_at=str(current.get("created_at") or created_at),
                updated_at=str(current.get("updated_at") or updated_at),
            )

    def create_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        """登记一个新会话并返回索引项。"""
        now = datetime.now().isoformat()
        self._get_session_dir(user_id, session_id)
        self._upsert_session_index(
            user_id,
            session_id,
            created_at=now,
            updated_at=now,
        )
        return next(
            item
            for item in self.list_sessions(user_id)
            if item["session_id"] == session_id
        )

    def is_session_empty(self, user_id: str, session_id: str) -> bool:
        """判断指定会话是否为空（无任何 user/assistant 消息）。"""
        jsonl_path = self._get_jsonl_path(user_id, session_id)
        if not jsonl_path.exists():
            return True

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("_type") == "metadata":
                        continue
                    if data.get("role") in {"user", "assistant"} and str(
                        data.get("content", "")
                    ).strip():
                        return False
        except Exception as e:
            warn("Store", f"判断会话 {session_id} 是否为空失败: {e}")
            return False

        return True

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定会话（含磁盘目录与索引条目）。"""
        sessions = self._read_index(user_id)
        new_sessions = [
            item for item in sessions if item.get("session_id") != session_id
        ]
        removed = len(new_sessions) != len(sessions)

        session_dir = self._get_sessions_dir(user_id) / session_id
        if session_dir.exists():
            try:
                shutil.rmtree(session_dir)
                removed = True
            except Exception as e:
                warn("Store", f"删除会话目录 {session_id} 失败: {e}")
                return False

        if removed:
            self._write_index(user_id, new_sessions)

        return removed

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """列出用户所有会话，并兼容旧版单会话数据。"""
        sessions = self._read_index(user_id)
        by_id = {
            str(item.get("session_id")): item
            for item in sessions
            if item.get("session_id")
        }

        sessions_dir = self._get_sessions_dir(user_id)
        for jsonl_path in sessions_dir.glob("*/session.jsonl"):
            session_id = jsonl_path.parent.name
            if session_id not in by_id:
                updated = datetime.fromtimestamp(jsonl_path.stat().st_mtime).isoformat()
                by_id[session_id] = {
                    "session_id": session_id,
                    "title": self._default_title(session_id, updated),
                    "created_at": updated,
                    "updated_at": updated,
                }

        legacy_path = self._get_legacy_jsonl_path(user_id)
        if legacy_path.exists() and user_id not in by_id:
            updated = datetime.fromtimestamp(legacy_path.stat().st_mtime).isoformat()
            by_id[user_id] = {
                "session_id": user_id,
                "title": "旧会话",
                "created_at": updated,
                "updated_at": updated,
            }

        result = list(by_id.values())
        result.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        if result != sessions:
            self._write_index(user_id, result)
        return result

    def load_session(self, user_id: str, session_id: Optional[str] = None) -> SessionData:
        """从 JSONL 加载会话"""
        session_id = session_id or user_id
        session = SessionData(session_id=session_id, user_id=user_id)
        jsonl_path = self._get_jsonl_path(user_id, session_id)

        if not jsonl_path.exists() and session_id == user_id:
            legacy_path = self._get_legacy_jsonl_path(user_id)
            if legacy_path.exists():
                jsonl_path = legacy_path

        if not jsonl_path.exists():
            self._upsert_session_index(user_id, session_id)
            return session

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        session.user_id = data.get("user_id") or user_id
                        session.created_at = datetime.fromisoformat(
                            data.get("created_at", datetime.now().isoformat())
                        )
                        session.updated_at = datetime.fromisoformat(
                            data.get("updated_at", datetime.now().isoformat())
                        )
                        session.last_consolidated = data.get("last_consolidated", 0)
                    else:
                        session.messages.append(data)
        except Exception as e:
            warn("Store", f"加载会话 {session_id} 失败: {e}")

        self._upsert_session_index(
            user_id,
            session_id,
            title=self._title_from_messages(session),
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
        )
        return session

    def save_session(self, session: SessionData):
        """保存会话到 JSONL"""
        user_id = session.user_id or session.session_id
        jsonl_path = self._get_jsonl_path(user_id, session.session_id)

        try:
            with open(jsonl_path, "w", encoding="utf-8") as f:
                metadata = {
                    "_type": "metadata",
                    "user_id": user_id,
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "last_consolidated": session.last_consolidated,
                }
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

                for msg in session.messages:
                    if not msg.get("message_id"):
                        msg["message_id"] = str(uuid.uuid4())
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

            self._upsert_session_index(
                user_id,
                session.session_id,
                title=self._title_from_messages(session),
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
            )
        except Exception as e:
            warn("Store", f"保存会话 {session.session_id} 失败: {e}")

    def get_memory_md_path(self, user_id: str) -> Path:
        """获取用户级 MEMORY.md 路径"""
        return self._get_visible_memory_dir(user_id) / "MEMORY.md"

    def get_history_md_path(self, user_id: str) -> Path:
        """获取用户级 HISTORY.md 路径"""
        return self._get_visible_memory_dir(user_id) / "HISTORY.md"

    def read_memory_md(self, user_id: str) -> str:
        """读取用户级 MEMORY.md 内容，如果不存在返回空字符串"""
        path = self.get_memory_md_path(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_memory_md(self, user_id: str, content: str):
        """写入用户级 MEMORY.md"""
        path = self.get_memory_md_path(user_id)
        path.write_text(content, encoding="utf-8")

    def read_history_md(self, user_id: str) -> str:
        """读取用户级 HISTORY.md 内容"""
        path = self.get_history_md_path(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "# Conversation History\n\n"

    def append_history_md(self, user_id: str, entry: str):
        """追加到用户级 HISTORY.md"""
        path = self.get_history_md_path(user_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")
