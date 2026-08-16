"""单元测试：覆盖 TravelAgent 解析、checkpoint、安全日志和 SubAgent 基础行为。"""

import asyncio
import json

import pytest

from industrial_runtime import safe_json, sanitize_payload
from subagents import SubAgentRunner
from tool_guard import ToolCallGuard
from tools.registry import ToolRegistry
from travel_agent import TravelAgent

from tests.conftest import FakeLLMClient, FakeMCPClient, FakeRuntime


pytestmark = [pytest.mark.unit, pytest.mark.full]


class FailingRegistry(ToolRegistry):
    """用于验证子代理本地工具执行失败后的恢复路径。"""
    def __init__(self):
        super().__init__()
        self.called = False

    def has(self, name: str) -> bool:
        return name == "local_fail"

    async def execute(self, name: str, params: dict):
        self.called = True
        raise RuntimeError("local failed")


def test_travel_agent_parses_tool_calls_from_supported_formats():
    """验证 Agent 能从纯 JSON、Markdown 代码块和混合文本中解析工具调用。"""
    agent = TravelAgent(enable_memory=False, mcp_client=FakeMCPClient())

    assert agent._parse_tool_call('{"tool":"lookup","arguments":{"q":"上海"}}') == {
        "tool": "lookup",
        "arguments": {"q": "上海"},
    }
    assert agent._parse_tool_call(
        '```json\n{"tool":"lookup","arguments":{"q":"北京"}}\n```'
    )["arguments"] == {"q": "北京"}
    assert agent._parse_tool_call(
        '先说明一下 {"tool":"lookup","arguments":{"q":"南京"}} 后续文本'
    )["tool"] == "lookup"
    assert agent._parse_tool_call("不是 JSON") is None


def test_checkpoint_state_validation_and_payload(fake_runtime):
    """验证 checkpoint 序列化、恢复节点选择以及非法状态回退逻辑。"""
    agent = TravelAgent(enable_memory=False, mcp_client=FakeMCPClient())
    agent.runtime = fake_runtime
    state = {
        "messages": [{"role": "user", "content": "hi"}],
        "tool_calls": [{"tool": "lookup", "arguments": {}}],
        "final_response": "",
        "request_id": "run-1",
        "retrieval_context": "history",
        "checkpoint_enabled": True,
    }

    payload = agent._checkpoint_payload(state, "tool")
    restored, node = agent._state_from_checkpoint(payload, event_queue=None)

    assert payload["next_node"] == "tool"
    assert node == "tool"
    assert restored["request_id"] == "run-1"
    assert agent._state_from_checkpoint({"next_node": "bad"}, None) == (None, "llm")
    assert agent._state_from_checkpoint({"messages": [], "tool_calls": []}, None) == (
        None,
        "llm",
    )


def test_agent_node_selection_cancel_and_prompt(fake_runtime):
    """验证节点流转、取消检测，以及系统提示词中技能和子代理信息的注入。"""
    agent = TravelAgent(
        enable_memory=False,
        skills_info="<skills><skill><name>weather</name></skill></skills>",
        mcp_client=FakeMCPClient(description="工具名: route\n描述: 路线"),
    )
    agent.runtime = fake_runtime

    assert agent.should_continue({"tool_calls": [{"tool": "route"}]}) == "tool"
    assert agent.should_continue({"tool_calls": [], "final_response": "ok"}) == "end"
    assert agent._should_continue_after_tool({"final_response": ""}) == "llm"
    assert "weather" in agent._build_system_prompt()
    assert "子代理工具" in agent._build_system_prompt()

    fake_runtime.cancelled.add("run-cancel")
    with pytest.raises(asyncio.CancelledError):
        agent._raise_if_run_cancelled("run-cancel")


def test_safe_json_and_sanitize_payload_protect_logs():
    """验证日志 payload 会脱敏敏感字段并截断过长内容，避免泄露或膨胀。"""
    payload = {
        "api_key": "secret",
        "nested": {"password": "pw", "text": "x" * 500},
        "items": list(range(40)),
    }

    sanitized = sanitize_payload(payload, limit=10)
    serialized = safe_json({"text": "x" * 2000}, limit=100)

    assert sanitized["api_key"] == "***"
    assert sanitized["nested"]["password"] == "***"
    assert sanitized["nested"]["text"].endswith("...(truncated)")
    assert len(sanitized["items"]) == 20
    assert json.loads(serialized)["_truncated"] is True


def test_tool_call_guard_duplicate_schema_and_retry_policy():
    """验证工具治理规则能识别重复调用、schema 缺参和可重试错误类型。"""
    guard = ToolCallGuard()
    chain = [
        {
            "tool": "lookup",
            "arguments": {"q": "上海"},
            "signature": ToolCallGuard.signature("lookup", {"q": "上海"}),
            "status": "success",
            "result_summary": "结果A",
        }
    ]

    second = guard.before_call(chain, "lookup", {"q": "上海"})
    third = guard.before_call(chain + [{"tool": "lookup", "arguments": {"q": "上海"}}], "lookup", {"q": "上海"})
    errors = guard.validate_arguments(
        "lookup",
        {},
        {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    )
    server_retry = guard.classify_exception(RuntimeError("HTTP 500 server error"))
    client_retry = guard.classify_exception(RuntimeError("HTTP 400 bad request"))

    assert second.action == "skip"
    assert third.action == "terminate"
    assert "结果A" in third.final_response
    assert errors == ["missing required q"]
    assert server_retry.retryable is True and server_retry.max_attempts == 3
    assert client_retry.retryable is False


@pytest.mark.asyncio
async def test_subagent_runner_descriptions_unknown_type_and_recursion_guard():
    """验证 SubAgent 描述、未知类型处理、工具调用解析和递归 task 防护。"""
    runner = SubAgentRunner(
        llm_client=FakeLLMClient(
            [
                '{"tool":"task","arguments":{"subagent_type":"general-purpose","task_description":"nested"}}',
                "递归被阻止后返回当前结果",
            ]
        ),
        model="test-model",
        mcp_client=FakeMCPClient(),
        tools=ToolRegistry(),
    )

    assert "general-purpose" in runner.get_available_subagents_desc()
    assert "travel-research" in runner.get_task_tool_description()
    assert "未知的子代理类型" in await runner.run("missing", "do something")
    assert runner._parse_tool_call('{"tool":"lookup","arguments":{}}')["tool"] == "lookup"

    result = await runner.run("general-purpose", "尝试递归 task", max_iterations=2)
    assert "递归被阻止" in result


@pytest.mark.asyncio
async def test_subagent_tool_failure_max_iterations_and_llm_error_paths():
    """验证 SubAgent 工具失败、最大轮次兜底和 LLM 异常会形成可读结果。"""
    failing_tools = FailingRegistry()
    runner = SubAgentRunner(
        llm_client=FakeLLMClient(
            [
                '{"tool":"local_fail","arguments":{"q":"x"}}',
                '{"tool":"remote_lookup","arguments":{"q":"y"}}',
            ]
        ),
        model="test-model",
        mcp_client=FakeMCPClient({"remote_lookup": "remote result"}),
        tools=failing_tools,
    )

    result = await runner.run("general-purpose", "工具失败后继续", max_iterations=2)

    assert failing_tools.called is True
    assert "最大迭代轮次" in result
    assert "remote_lookup" in result

    error_runner = SubAgentRunner(
        llm_client=FakeLLMClient([RuntimeError("llm down")]),
        model="test-model",
        mcp_client=FakeMCPClient(),
        tools=ToolRegistry(),
    )

    assert "执行失败" in await error_runner.run("general-purpose", "LLM 异常")


@pytest.mark.asyncio
async def test_subagent_prompt_local_tool_filter_and_mcp_execution():
    """验证子代理提示词会过滤 task 工具，并能走 MCP 工具执行路径。"""
    registry = ToolRegistry()
    runner = SubAgentRunner(
        llm_client=FakeLLMClient([]),
        model="test-model",
        mcp_client=FakeMCPClient({"lookup": "mcp ok"}, description="工具名: lookup\n描述: 查找"),
        tools=registry,
    )

    class TaskLikeTool:
        name = "task"
        description = "不应出现在子代理本地工具说明中"

    class LocalTool:
        name = "local"
        description = "本地可用"

    registry.register(TaskLikeTool())  # type: ignore[arg-type]
    registry.register(LocalTool())  # type: ignore[arg-type]

    prompt = runner._build_subagent_system_prompt(runner.subagent_configs["general-purpose"])
    mcp_result = await runner._execute_tool("lookup", {"q": "上海"})

    assert "工具名: lookup" in prompt
    assert "本地可用" in prompt
    assert "不应出现在" not in prompt
    assert mcp_result == "mcp ok"
