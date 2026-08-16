"""工具调用治理：重复调用、参数校验、失败计数和重试分类。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class GuardDecision:
    """工具执行前的治理决策。"""

    action: str
    message: str = ""
    final_response: str = ""
    signature: str = ""
    attempt: int = 1
    last_result_summary: str = ""


@dataclass
class RetryDecision:
    """工具失败后的重试决策。"""

    error_type: str
    retryable: bool
    max_attempts: int


class ToolCallGuard:
    """集中处理主 Agent 工具调用的硬规则。"""

    DUPLICATE_HINT = "该工具已用相同参数调用过，请基于已有工具结果回答，或更换参数/工具继续。"

    TASK_SCHEMA = {
        "type": "object",
        "properties": {
            "subagent_type": {"type": "string"},
            "task_description": {"type": "string", "minLength": 1},
        },
        "required": ["subagent_type", "task_description"],
    }

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def __init__(
        self,
        duplicate_warning_attempt: int = 2,
        duplicate_stop_attempt: int = 3,
        failure_stop_count: int = 3,
    ) -> None:
        self.duplicate_warning_attempt = duplicate_warning_attempt
        self.duplicate_stop_attempt = duplicate_stop_attempt
        self.failure_stop_count = failure_stop_count

    @classmethod
    def canonical_json(cls, arguments: dict[str, Any]) -> str:
        """稳定序列化参数，避免 key 顺序不同导致重复调用漏判。"""
        normalized = cls._normalize_value(arguments)
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def signature(cls, tool_name: str, arguments: dict[str, Any]) -> str:
        """生成工具调用签名。"""
        return f"{tool_name}:{cls.canonical_json(arguments)}"

    def before_call(
        self,
        tool_call_chain: list[dict[str, Any]],
        tool_name: str,
        arguments: dict[str, Any],
    ) -> GuardDecision:
        """执行工具前检查是否重复调用。"""
        signature = self.signature(tool_name, arguments)
        same_records = [
            record for record in tool_call_chain
            if self._record_signature(record) == signature
        ]
        attempt = len(same_records) + 1
        last_result_summary = self._last_result_summary(same_records)

        if attempt >= self.duplicate_stop_attempt:
            return GuardDecision(
                action="terminate",
                signature=signature,
                attempt=attempt,
                last_result_summary=last_result_summary,
                final_response=self._build_duplicate_stop_response(
                    tool_name,
                    arguments,
                    last_result_summary,
                ),
            )
        if attempt >= self.duplicate_warning_attempt:
            return GuardDecision(
                action="skip",
                message=self.DUPLICATE_HINT,
                signature=signature,
                attempt=attempt,
                last_result_summary=last_result_summary,
            )
        return GuardDecision(action="proceed", signature=signature, attempt=attempt)

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        schema: dict[str, Any] | None,
    ) -> list[str]:
        """按工具注册时的 schema 做执行前硬校验。"""
        if not schema:
            return []
        if tool_name == "task":
            schema = self.TASK_SCHEMA
        if not isinstance(arguments, dict):
            return [f"parameters must be an object, got {type(arguments).__name__}"]
        return self._validate(arguments, {**schema, "type": "object"}, "")

    def classify_exception(self, exc: Exception) -> RetryDecision:
        """把异常归类为重试策略，避免所有错误都盲目重试。"""
        text = str(exc).lower()
        if self._contains_http_status(text, range(400, 500)) or "invalid parameters" in text:
            return RetryDecision("client", False, 1)
        if "missing required" in text or "参数" in text:
            return RetryDecision("validation", False, 1)
        if self._contains_http_status(text, range(500, 600)) or "server" in text or "服务器" in text:
            return RetryDecision("server", True, 3)
        if "timeout" in text or "timed out" in text or "超时" in text:
            return RetryDecision("timeout", True, 3)
        if any(token in text for token in ["重连失败", "连接断开", "连接异常", "connection", "transport"]):
            return RetryDecision("mcp_connection", True, 3)
        return RetryDecision("other", True, 2)

    def should_stop_after_failure(self, tool_call_chain: list[dict[str, Any]]) -> bool:
        """连续失败达到阈值时提前终止 Loop。"""
        failures = 0
        for record in reversed(tool_call_chain):
            if record.get("status") == "failed":
                failures += 1
                if failures >= self.failure_stop_count:
                    return True
            elif record.get("status") not in {"skipped", "blocked"}:
                return False
        return False

    def build_failure_stop_response(self, tool_call_chain: list[dict[str, Any]]) -> str:
        """生成连续工具失败后的兜底说明。"""
        failed = [
            record for record in tool_call_chain
            if record.get("status") == "failed"
        ][-self.failure_stop_count:]
        lines = ["连续多次工具调用失败，已提前停止本轮任务，避免继续空转。", "", "失败摘要："]
        for record in failed:
            tool = record.get("tool") or "unknown"
            err = record.get("error") or record.get("result_summary") or "未知错误"
            lines.append(f"- {tool}: {err}")
        return "\n".join(lines)

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        """对参数做轻量归一化，保留语义同时提升重复识别稳定性。"""
        if isinstance(value, dict):
            return {
                str(key): cls._normalize_value(val)
                for key, val in value.items()
                if val is not None
            }
        if isinstance(value, list):
            return [cls._normalize_value(item) for item in value]
        if isinstance(value, str):
            return value.strip()
        return value

    @classmethod
    def _record_signature(cls, record: dict[str, Any]) -> str:
        if record.get("signature"):
            return str(record["signature"])
        return cls.signature(str(record.get("tool", "")), record.get("arguments") or {})

    @staticmethod
    def _last_result_summary(records: list[dict[str, Any]]) -> str:
        for record in reversed(records):
            summary = record.get("result_summary")
            if summary:
                return str(summary)
        return ""

    def _build_duplicate_stop_response(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        last_result_summary: str,
    ) -> str:
        args = self.canonical_json(arguments)
        summary = last_result_summary or "暂无可用结果摘要。"
        return (
            "已检测到连续重复调用相同工具和参数，已提前停止本轮工具调用。\n\n"
            f"- 工具: {tool_name}\n"
            f"- 参数: {args}\n"
            f"- 已有结果摘要: {summary}\n\n"
            "不能继续重复调用的原因：相同输入不会带来新增信息，继续调用可能导致任务空转。"
        )

    @classmethod
    def _resolve_type(cls, raw_type: Any) -> str | None:
        if isinstance(raw_type, list):
            for item in raw_type:
                if item != "null":
                    return item
            return None
        return raw_type

    @classmethod
    def _validate(cls, value: Any, schema: dict[str, Any], path: str) -> list[str]:
        raw_type = schema.get("type")
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get("nullable", False)
        schema_type = cls._resolve_type(raw_type)
        label = path or "parameter"

        if nullable and value is None:
            return []
        if schema_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return [f"{label} should be integer"]
        if schema_type == "number" and (
            not isinstance(value, cls._TYPE_MAP["number"]) or isinstance(value, bool)
        ):
            return [f"{label} should be number"]
        if (
            schema_type in cls._TYPE_MAP
            and schema_type not in {"integer", "number"}
            and not isinstance(value, cls._TYPE_MAP[schema_type])
        ):
            return [f"{label} should be {schema_type}"]

        errors: list[str] = []
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if schema_type in {"integer", "number"}:
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if schema_type == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if schema_type == "object":
            props = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in value:
                    errors.append(f"missing required {path + '.' + key if path else key}")
            for key, nested_value in value.items():
                if key in props:
                    nested_path = path + "." + key if path else key
                    errors.extend(cls._validate(nested_value, props[key], nested_path))
        if schema_type == "array" and "items" in schema:
            for index, item in enumerate(value):
                nested_path = f"{path}[{index}]" if path else f"[{index}]"
                errors.extend(cls._validate(item, schema["items"], nested_path))
        return errors

    @staticmethod
    def _contains_http_status(text: str, status_range: range) -> bool:
        for match in re.finditer(r"\b([1-5]\d{2})\b", text):
            if int(match.group(1)) in status_range:
                return True
        return False
