"""Travel Agent 的 Langfuse 可观测性辅助模块。"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL, LANGFUSE_HOST, LANGFUSE_TRACING_ENABLED
from logger import info, warn


TRACE_NAME = "travel-agent-chat"
_initialized = False
_configured = False
_warned_missing_config = False


def _normalize_langfuse_env() -> None:

    base_url = LANGFUSE_BASE_URL
    host = LANGFUSE_HOST
    if host and not base_url:
        os.environ["LANGFUSE_BASE_URL"] = host
    elif base_url and not host:
        os.environ["LANGFUSE_HOST"] = base_url


def _has_langfuse_keys() -> bool:
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def init_langfuse():
    """初始化 Langfuse（仅一次），未配置时保持应用正常运行。"""
    global _initialized, _configured, _warned_missing_config

    if _initialized:
        return get_langfuse_client()

    _normalize_langfuse_env()
    _configured = _has_langfuse_keys()

    try:
        from langfuse import Langfuse, get_client
    except Exception as exc:
        if not _warned_missing_config:
            warn("Langfuse", f"SDK 不可用，已跳过监控: {exc}")
            _warned_missing_config = True
        _initialized = True
        return None

    if _configured:
        client = get_client()
    else:
        os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
        Langfuse(public_key="disabled", secret_key="disabled", tracing_enabled=False)
        client = get_client()
        if not _warned_missing_config:
            warn("Langfuse", "未配置 PUBLIC_KEY/SECRET_KEY，监控保持关闭")
            _warned_missing_config = True

    _initialized = True
    return client


def get_langfuse_client():
    if not _initialized:
        return init_langfuse()

    try:
        from langfuse import get_client

        return get_client()
    except Exception:
        return None


def is_langfuse_enabled() -> bool:
    if not _initialized:
        init_langfuse()
    return _configured and LANGFUSE_TRACING_ENABLED.lower() != "false"


def get_langfuse_metadata(
    session_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    component: str | None = None,
    operation: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "app": "travel-agent",
    }
    if session_id:
        metadata["langfuse_session_id"] = str(session_id)
    if user_id:
        metadata["langfuse_user_id"] = str(user_id)
    if request_id:
        metadata["request_id"] = str(request_id)
    if component:
        metadata["component"] = component
    if operation:
        metadata["operation"] = operation
    if extra:
        metadata.update(extra)
    return metadata


def _propagation_metadata(
    request_id: str | None = None,
    component: str | None = None,
    operation: str | None = None,
) -> dict[str, str]:
    metadata: dict[str, str] = {"app": "travel-agent"}
    if request_id:
        metadata["request_id"] = str(request_id)[:500]
    if component:
        metadata["component"] = str(component)[:500]
    if operation:
        metadata["operation"] = str(operation)[:500]
    return metadata


@contextmanager
def langfuse_request_trace(
    *,
    session_id: str | None,
    user_id: str | None,
    request_id: str | None,
    input_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """创建请求级别的 Langfuse 观测，并传播 trace 属性。"""
    if not is_langfuse_enabled():
        yield None
        return

    try:
        from langfuse import propagate_attributes

        client = get_langfuse_client()
        if client is None:
            yield None
            return

        trace_metadata = get_langfuse_metadata(
            session_id=session_id,
            user_id=user_id,
            request_id=request_id,
            component="travel_agent",
            operation="chat",
            extra=metadata,
        )
        propagation = propagate_attributes(
            user_id=str(user_id) if user_id else None,
            session_id=str(session_id) if session_id else None,
            metadata=_propagation_metadata(request_id, "travel_agent", "chat"),
            tags=["travel-agent"],
            trace_name=TRACE_NAME,
        )
        with client.start_as_current_observation(
            name=TRACE_NAME,
            as_type="agent",
            input=input_data,
            metadata=trace_metadata,
        ) as observation:
            with propagation:
                yield observation
    except Exception as exc:
        warn("Langfuse", f"请求 trace 创建失败，继续执行: {exc}")
        with nullcontext():
            yield None


def update_langfuse_observation(observation: Any | None, **kwargs: Any) -> None:
    if observation is None:
        return
    try:
        observation.update(**kwargs)
    except Exception:
        pass


@contextmanager
def langfuse_tool_span(
    *,
    tool_name: str,
    arguments: Any = None,
    session_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
) -> Iterator[Any | None]:
    """为单次工具调用创建独立的 Langfuse span，自动采集工具名、参数与耗时。"""
    if not is_langfuse_enabled():
        yield None
        return

    client = get_langfuse_client()
    if client is None:
        yield None
        return

    try:
        metadata = get_langfuse_metadata(
            session_id=session_id,
            user_id=user_id,
            request_id=request_id,
            component="travel_agent",
            operation="tool_call",
            extra={"tool_name": str(tool_name)},
        )
        with client.start_as_current_observation(
            name=f"tool:{tool_name}",
            as_type="tool",
            input={"tool": tool_name, "arguments": arguments},
            metadata=metadata,
        ) as span:
            yield span
    except Exception as exc:
        warn("Langfuse", f"工具 span 创建失败，继续执行: {exc}")
        yield None


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass
