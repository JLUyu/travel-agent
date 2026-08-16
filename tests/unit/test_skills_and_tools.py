"""单元测试：覆盖本地工具注册/校验、文件读取、澄清工具、Shell 工具和 Skill 加载。"""

import json
import os
from pathlib import Path

import pytest

from skills_loader import SkillsLoader
from tools.base import Tool
from tools.clarification import AskClarificationTool
from tools.filesystem import ReadFileTool
from tools.registry import ToolRegistry
from tools.shell import ExecTool


pytestmark = [pytest.mark.unit, pytest.mark.full]


class DemoTool(Tool):
    """用于测试 ToolRegistry 参数转换和校验逻辑的最小工具。"""
    @property
    def name(self):
        return "demo"

    @property
    def description(self):
        return "demo tool"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1},
                "enabled": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["count"],
        }

    async def execute(self, **kwargs):
        return kwargs


class NonObjectSchemaTool(DemoTool):
    """用于验证非 object schema 会触发显式校验错误。"""
    @property
    def parameters(self):
        return {"type": "string"}


class ErrorTool(DemoTool):
    """用于验证注册表会给工具错误结果追加提示。"""
    @property
    def name(self):
        return "error_tool"

    async def execute(self, **kwargs):
        return "Error: failed"


class RaisingTool(DemoTool):
    """用于验证注册表会捕获工具执行异常。"""
    @property
    def name(self):
        return "raising_tool"

    async def execute(self, **kwargs):
        raise RuntimeError("boom")


class ReadOnlyDemoTool(DemoTool):
    """用于验证只读工具的并发安全默认值。"""
    @property
    def read_only(self):
        return True


class ComplexSchemaTool(DemoTool):
    """用于覆盖复杂 JSON Schema 转换和校验分支。"""
    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 2, "maxLength": 4},
                "mode": {"type": ["null", "string"], "enum": ["fast", None]},
                "ratio": {"type": "number", "minimum": 1.5, "maximum": 3.0},
                "nested": {
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean"}},
                    "required": ["enabled"],
                },
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["name", "nested"],
        }


def test_tool_cast_validate_and_registry_execution():
    """验证工具参数类型转换、必填/范围校验以及未知工具错误。"""
    registry = ToolRegistry()
    registry.register(DemoTool())

    tool, params, error = registry.prepare_call(
        "demo", {"count": "2", "enabled": "yes", "tags": [1, "b"]}
    )

    assert error is None
    assert params == {"count": 2, "enabled": True, "tags": ["1", "b"]}
    assert tool.validate_params({"count": 0}) == ["count must be >= 1"]
    assert registry.prepare_call("missing", {})[2].startswith("Error: Tool 'missing'")


def test_tool_schema_casting_validation_and_metadata_branches():
    """验证 Tool 基类的嵌套转换、校验错误和 schema 导出分支。"""
    tool = ComplexSchemaTool()

    casted = tool.cast_params(
        {
            "name": 123,
            "mode": None,
            "ratio": "2.5",
            "nested": {"enabled": "no"},
            "items": ["1", "bad"],
            "unknown": "kept",
        }
    )

    assert casted["name"] == "123"
    assert casted["ratio"] == 2.5
    assert casted["nested"]["enabled"] is False
    assert casted["items"] == [1, "bad"]
    assert casted["unknown"] == "kept"
    assert tool.validate_params(casted) == ["items[1] should be integer"]
    assert tool.validate_params({"name": "x", "ratio": 4, "nested": {}}) == [
        "name must be at least 2 chars",
        "ratio must be <= 3.0",
        "missing required nested.enabled",
    ]
    assert tool.validate_params({"name": "abcde", "mode": "slow", "nested": {"enabled": True}}) == [
        "name must be at most 4 chars",
        "mode must be one of ['fast', None]",
    ]
    assert tool.validate_params("bad") == ["parameters must be an object, got str"]
    with pytest.raises(ValueError):
        NonObjectSchemaTool().validate_params({})
    assert ReadOnlyDemoTool().concurrency_safe is True
    assert tool.to_schema()["function"]["name"] == "demo"


@pytest.mark.asyncio
async def test_read_file_tool_handles_pagination_and_file_types(tmp_path):
    """验证文件读取工具的分页、越界、二进制文件和目录越权保护。"""
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    text_file = allowed / "notes.txt"
    text_file.write_text("a\nb\nc\n", encoding="utf-8")
    binary_file = allowed / "image.bin"
    binary_file.write_bytes(b"\xff\x00\x01")

    tool = ReadFileTool(workspace=allowed, allowed_dir=allowed)

    assert "1| a\n2| b" in await tool.execute(path="notes.txt", offset=1, limit=2)
    assert "Use offset=3" in await tool.execute(path="notes.txt", offset=1, limit=2)
    assert "offset 99 is beyond end" in await tool.execute(path="notes.txt", offset=99)
    assert "Cannot read binary file" in await tool.execute(path="image.bin")

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    assert "outside allowed directory" in await tool.execute(path=str(outside))


@pytest.mark.asyncio
async def test_read_file_tool_error_empty_and_trimming_branches(tmp_path):
    """验证读取文件工具的缺失路径、目录、空文件、负 offset 和超长截断分支。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    empty_file = workspace / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    long_file = workspace / "long.txt"
    long_file.write_text("\n".join(["x" * 1000 for _ in range(200)]), encoding="utf-8")
    tool = ReadFileTool(workspace=workspace, allowed_dir=workspace)

    assert "Unknown path" in await tool.execute(path="")
    assert "File not found" in await tool.execute(path="missing.txt")
    assert "Not a file" in await tool.execute(path=".")
    assert "Empty file" in await tool.execute(path="empty.txt")

    negative_offset = await tool.execute(path="long.txt", offset=-10, limit=1)
    trimmed = await tool.execute(path="long.txt", offset=1, limit=200)

    assert negative_offset.startswith("1| ")
    assert "Use offset=2" in negative_offset
    assert len(trimmed) <= ReadFileTool._MAX_CHARS + 100
    assert "Use offset=" in trimmed


@pytest.mark.asyncio
async def test_ask_clarification_returns_structured_json():
    """验证澄清工具返回可解析的结构化 JSON，方便前端展示问题和选项。"""
    result = await AskClarificationTool().execute(
        question="你想哪天出发？",
        clarification_type="missing_info",
        context="缺少日期",
        options=["今天", "明天"],
    )
    payload = json.loads(result)

    assert payload["question"] == "你想哪天出发？"
    assert payload["options"] == ["今天", "明天"]


@pytest.mark.asyncio
async def test_exec_tool_direct_success_guard_and_timeout(tmp_path):
    """验证 Shell 工具的正常执行、安全拦截和超时处理。"""
    tool = ExecTool(working_dir=str(tmp_path), timeout=2, use_sandbox=False)

    result = await tool.execute("printf ok")
    assert "ok" in result
    assert "Exit code: 0" in result
    assert "blocked by safety guard" in await tool.execute("rm -rf /tmp/something")

    timeout = await tool.execute(
        "python -c \"import time; time.sleep(2)\"",
        timeout=1,
    )
    assert "timed out" in timeout


@pytest.mark.asyncio
async def test_exec_tool_guard_format_sandbox_and_config_branches(tmp_path, monkeypatch):
    """验证 Shell 工具的 allowlist、路径穿越、沙箱缺失、配置查找和输出格式分支。"""
    config_path = tmp_path / ".srt-settings.json"
    config_path.write_text("{}", encoding="utf-8")
    tool = ExecTool(
        working_dir=str(tmp_path),
        use_sandbox=True,
        allow_patterns=[r"^echo"],
        restrict_to_workspace=True,
    )

    assert tool._find_sandbox_config() == str(config_path.resolve())
    assert "not in allowlist" in tool._guard_command("python -V", str(tmp_path))
    assert "path traversal" in tool._guard_command("echo ../secret", str(tmp_path))
    assert tool._guard_command("echo ok", str(tmp_path)) is None

    sandboxless = ExecTool(working_dir=str(tmp_path / "missing"), use_sandbox=True)
    assert "no .srt-settings.json" in await sandboxless.execute("echo ok")

    formatted = tool._format_output(b"out", b"err", 7, sandboxed=True)
    huge = tool._format_output(b"x" * (ExecTool._MAX_OUTPUT + 100), b"", 0, sandboxed=False)

    assert "[Sandbox]" in formatted
    assert "STDERR" in formatted
    assert "Exit code: 7" in formatted
    assert "chars truncated" in huge

    async def fail_shell(*args, **kwargs):
        raise RuntimeError("spawn failed")

    direct = ExecTool(working_dir=str(tmp_path), use_sandbox=False)
    monkeypatch.setattr("asyncio.create_subprocess_shell", fail_shell)
    assert "spawn failed" in await direct.execute("printf ok")


@pytest.mark.asyncio
async def test_registry_execute_success_error_and_exception_paths():
    """验证 ToolRegistry 的注册、注销、定义导出、错误提示和异常捕获分支。"""
    registry = ToolRegistry()
    demo = DemoTool()
    registry.register(demo)
    registry.register(ErrorTool())
    registry.register(RaisingTool())

    assert registry.has("demo")
    assert "demo" in registry
    assert len(registry) == 3
    assert registry.get("demo") is demo
    assert registry.get_definitions()[0]["type"] == "function"
    assert await registry.execute("demo", {"count": "3"}) == {"count": 3}
    assert "try a different approach" in await registry.execute("missing", {})
    assert "try a different approach" in await registry.execute("error_tool", {"count": 1})
    assert "Error executing raising_tool: boom" in await registry.execute("raising_tool", {"count": 1})

    registry.unregister("demo")
    assert not registry.has("demo")


def test_skills_loader_discovers_metadata_requirements_and_context(tmp_path, monkeypatch):
    """验证 Skill 扫描、metadata 解析、环境依赖过滤和上下文拼接。"""
    skills_dir = tmp_path / "skills"
    (skills_dir / "weather").mkdir(parents=True)
    (skills_dir / "weather" / "SKILL.md").write_text(
        """---
description: 查询天气
metadata: {"agent":{"always":true}}
---
# Weather Skill
使用天气工具。""",
        encoding="utf-8",
    )
    (skills_dir / "blocked").mkdir()
    (skills_dir / "blocked" / "SKILL.md").write_text(
        """---
description: 不可用
metadata: {"agent":{"requires":{"env":["MISSING_TEST_ENV"]}}}
---
# Blocked""",
        encoding="utf-8",
    )
    monkeypatch.delenv("MISSING_TEST_ENV", raising=False)

    loader = SkillsLoader(skills_dir)

    assert {s["name"] for s in loader.list_skills(filter_unavailable=False)} == {
        "weather",
        "blocked",
    }
    assert [s["name"] for s in loader.list_skills(filter_unavailable=True)] == ["weather"]
    assert loader.get_always_skills() == ["weather"]
    assert "### Skill: weather" in loader.load_skills_for_context(["weather"])
    assert '<skill available="false">' in loader.build_skills_summary()
