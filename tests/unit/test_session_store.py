"""单元测试：覆盖 SessionStore 的会话持久化、索引、兼容路径和用户级记忆文件。"""

import json
from pathlib import Path

import pytest

from session_store import SessionData


pytestmark = [pytest.mark.unit, pytest.mark.full]


def test_session_store_round_trips_jsonl_and_index(session_store):
    """验证 JSONL 会话保存/加载后消息、索引标题和归档位置保持一致。"""
    session = SessionData(session_id="session-1", user_id="user-1")
    session.add_message("user", "上海三日游")
    session.add_message("assistant", "可以安排外滩、豫园和博物馆。")
    session.update_last_consolidated(1)

    session_store.save_session(session)
    loaded = session_store.load_session("user-1", "session-1")

    assert loaded.user_id == "user-1"
    assert loaded.session_id == "session-1"
    assert loaded.last_consolidated == 1
    assert [m["role"] for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[0]["content"] == "上海三日游"

    sessions = session_store.list_sessions("user-1")
    assert sessions[0]["session_id"] == "session-1"
    assert sessions[0]["title"] == "上海三日游"
    assert not session_store.is_session_empty("user-1", "session-1")


def test_empty_and_deleted_sessions_are_handled(session_store):
    """验证空会话判断、删除成功路径和重复删除的失败返回。"""
    session_store.create_session("user-2", "empty-session")

    assert session_store.is_session_empty("user-2", "empty-session")
    assert session_store.delete_session("user-2", "empty-session")
    assert session_store.list_sessions("user-2") == []
    assert not session_store.delete_session("user-2", "empty-session")


def test_load_session_tolerates_corrupt_jsonl(session_store):
    """验证 JSONL 中出现损坏行时，仍能加载前面有效消息。"""
    session = SessionData(session_id="session-corrupt", user_id="user-3")
    session.add_message("user", "第一条能读到")
    session_store.save_session(session)

    jsonl_path = session_store._get_jsonl_path("user-3", "session-corrupt")
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")

    loaded = session_store.load_session("user-3", "session-corrupt")

    assert loaded.messages[0]["content"] == "第一条能读到"


def test_legacy_session_path_is_listed_and_loaded(session_store):
    """验证旧版单文件会话路径仍能被列表和加载逻辑兼容。"""
    legacy_path = session_store._get_legacy_jsonl_path("legacy-user")
    metadata = {
        "_type": "metadata",
        "user_id": "legacy-user",
        "session_id": "legacy-user",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:01:00",
        "last_consolidated": 0,
    }
    message = {"role": "user", "content": "旧会话", "message_id": "m1"}
    legacy_path.write_text(
        json.dumps(metadata, ensure_ascii=False) + "\n"
        + json.dumps(message, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sessions = session_store.list_sessions("legacy-user")
    loaded = session_store.load_session("legacy-user", "legacy-user")

    assert sessions[0]["session_id"] == "legacy-user"
    assert loaded.messages[0]["content"] == "旧会话"


def test_memory_and_history_files_are_user_scoped(session_store):
    """验证 memory.md 和 history.md 按 user_id 隔离，避免跨用户读取。"""
    session_store.write_memory_md("user-4", "喜欢高铁出行")
    session_store.append_history_md("user-4", "[2026-01-01 10:00] 讨论南京旅行")

    assert session_store.read_memory_md("user-4") == "喜欢高铁出行"
    assert "南京旅行" in session_store.read_history_md("user-4")
    assert session_store.read_memory_md("other-user") == ""
