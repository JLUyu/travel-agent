"""集成测试：覆盖 TravelAgent.chat_stream 的直接回答、工具调用、澄清、子代理和恢复流程。"""

import json
from types import SimpleNamespace

import pytest
import travel_agent


pytestmark = [pytest.mark.integration, pytest.mark.full]


async def collect_events(agent, message: str, run_id: str = "run-stream"):
    """收集 chat_stream 产生的全部事件，便于断言事件顺序和内容。"""
    events = []
    async for event in agent.chat_stream(message, run_id=run_id):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_agent_stream_direct_answer_persists_messages(agent_factory, fake_runtime):
    """验证无工具调用时直接流式输出最终回答，并持久化用户和助手消息。"""
    agent = agent_factory(["这是最终回答"])

    events = await collect_events(agent, "你好")
    loaded = agent.session_store.load_session(agent.user_id, agent.session_id)

    assert any(event["type"] == "status" for event in events)
    assert "".join(event.get("content", "") for event in events if event["type"] == "content") == "这是最终回答"
    assert events[-1]["type"] == "done"
    assert [msg["role"] for msg in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[-1]["content"] == "这是最终回答"
    assert fake_runtime.runs["run-stream"]["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_stream_calls_mcp_tool_then_summarizes(agent_factory):
    """验证 Agent 先调用 MCP 工具，再基于工具结果生成总结。"""
    tool_call = json.dumps(
        {"tool": "lookup", "arguments": {"query": "上海天气"}},
        ensure_ascii=False,
    )
    agent = agent_factory([tool_call, "工具返回后总结"], {"lookup": "晴天，25度"})

    events = await collect_events(agent, "查天气", run_id="run-tool")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content")

    assert agent.mcp_client.calls == [("lookup", {"query": "上海天气"})]
    assert any(event["type"] == "tool_start" and event["tool"] == "lookup" for event in events)
    assert any(event["type"] == "tool_result" and event["status"] == "success" for event in events)
    assert content == "工具返回后总结"


@pytest.mark.asyncio
async def test_agent_stream_handles_tool_failure_with_llm_summary(agent_factory):
    """验证工具失败时仍会把失败信息交给 LLM 生成面向用户的说明。"""
    tool_call = '{"tool":"lookup","arguments":{"query":"bad"}}'
    agent = agent_factory([tool_call, "工具失败，但我会说明限制"], {"lookup": RuntimeError("断线")})

    events = await collect_events(agent, "查失败场景", run_id="run-fail-tool")

    assert any(event["type"] == "tool_result" and event["status"] == "failed" for event in events)
    assert "工具失败" in "".join(
        event.get("content", "") for event in events if event["type"] == "content"
    )


@pytest.mark.asyncio
async def test_agent_stream_skips_second_duplicate_tool_call(agent_factory):
    """验证第二次同工具同参数会被拦截，并把修正提示交回 LLM。"""
    tool_call = '{"tool":"lookup","arguments":{"query":"上海天气"}}'
    agent = agent_factory([tool_call, tool_call, "基于已有结果回答"], {"lookup": "晴天"})

    events = await collect_events(agent, "查天气", run_id="run-duplicate-skip")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content")

    assert agent.mcp_client.calls == [("lookup", {"query": "上海天气"})]
    assert any(event["type"] == "tool_result" and event["status"] == "skipped" for event in events)
    assert content == "基于已有结果回答"


@pytest.mark.asyncio
async def test_agent_stream_stops_third_duplicate_tool_call(agent_factory):
    """验证第三次同工具同参数会提前终止，避免重复工具调用空转。"""
    tool_call = '{"tool":"lookup","arguments":{"query":"上海天气"}}'
    agent = agent_factory([tool_call, tool_call, tool_call], {"lookup": "晴天"})

    events = await collect_events(agent, "查天气", run_id="run-duplicate-stop")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content")

    assert agent.mcp_client.calls == [("lookup", {"query": "上海天气"})]
    assert "连续重复调用相同工具和参数" in content


@pytest.mark.asyncio
async def test_agent_stream_validates_mcp_arguments_before_call(agent_factory):
    """验证 MCP 工具会按注册 schema 做调用前硬校验，缺参时不触发真实调用。"""
    bad_call = '{"tool":"lookup","arguments":{}}'
    good_call = '{"tool":"lookup","arguments":{"query":"上海天气"}}'
    agent = agent_factory([bad_call, good_call, "参数修正后总结"], {"lookup": "晴天"})
    agent.mcp_client.tools["lookup"] = SimpleNamespace(
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    )

    events = await collect_events(agent, "查天气", run_id="run-mcp-validation")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content")

    assert agent.mcp_client.calls == [("lookup", {"query": "上海天气"})]
    assert any("参数校验失败" in event.get("message", "") for event in events)
    assert content == "参数修正后总结"


@pytest.mark.asyncio
async def test_agent_stream_stops_after_three_tool_failures(agent_factory):
    """验证连续三次工具失败会提前终止本轮 Loop。"""
    calls = [
        '{"tool":"lookup","arguments":{"query":"a"}}',
        '{"tool":"lookup","arguments":{"query":"b"}}',
        '{"tool":"lookup","arguments":{"query":"c"}}',
    ]
    agent = agent_factory(calls, {"lookup": RuntimeError("HTTP 400 bad request")})

    events = await collect_events(agent, "查失败", run_id="run-three-failures")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content")

    assert len(agent.mcp_client.calls) == 3
    assert "连续多次工具调用失败" in content


@pytest.mark.asyncio
async def test_agent_stream_retries_mcp_server_error(agent_factory, monkeypatch):
    """验证 5xx/服务器错误会按指数退避策略重试，成功后继续总结。"""
    attempts = {"count": 0}

    async def fake_sleep(_delay):
        return None

    def flaky_lookup(_arguments):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("HTTP 500 server error")
        return "重试后成功"

    monkeypatch.setattr(travel_agent.asyncio, "sleep", fake_sleep)
    tool_call = '{"tool":"lookup","arguments":{"query":"上海天气"}}'
    agent = agent_factory([tool_call, "重试后总结"], {"lookup": flaky_lookup})

    events = await collect_events(agent, "查天气", run_id="run-mcp-retry")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content")

    assert attempts["count"] == 3
    assert len(agent.mcp_client.calls) == 3
    assert content == "重试后总结"


@pytest.mark.asyncio
async def test_agent_stream_ask_clarification_stops_current_turn(agent_factory):
    """验证澄清工具会中断当前轮次，并直接返回问题与候选项。"""
    tool_call = json.dumps(
        {
            "tool": "ask_clarification",
            "arguments": {
                "question": "你想哪天出发？",
                "context": "缺少出发日期",
                "options": ["今天", "明天"],
            },
        },
        ensure_ascii=False,
    )
    agent = agent_factory([tool_call])

    events = await collect_events(agent, "帮我买票", run_id="run-clarify")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content")

    assert "缺少出发日期" in content
    assert "你想哪天出发？" in content
    assert agent.mcp_client.calls == []


@pytest.mark.asyncio
async def test_agent_stream_delegates_to_subagent(agent_factory):
    """验证主 Agent 能通过 task 工具委托 SubAgent，并继续汇总子代理结果。"""
    task_call = json.dumps(
        {
            "tool": "task",
            "arguments": {
                "subagent_type": "travel-research",
                "task_description": "调研上海亲子景点",
            },
        },
        ensure_ascii=False,
    )
    agent = agent_factory([task_call, "子代理结果：推荐科技馆", "主代理总结：推荐科技馆"])

    events = await collect_events(agent, "请调研上海亲子景点", run_id="run-subagent")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content")

    assert "主代理总结" in content
    assert any(event["type"] == "tool_start" and event["tool"] == "task" for event in events)


@pytest.mark.asyncio
async def test_agent_resume_uses_checkpoint_without_duplicate_user_message(agent_factory, fake_runtime):
    """验证从 checkpoint 恢复时不会重复写入同一条用户消息。"""
    agent = agent_factory(["恢复后的最终回答"])
    run_id = "run-resume"
    fake_runtime.checkpoints[run_id] = [
        {
            "version": 1,
            "run_id": run_id,
            "next_node": "llm",
            "messages": [{"role": "user", "content": "恢复问题"}],
            "tool_calls": [],
            "final_response": "",
            "retrieval_context": "",
        }
    ]

    events = []
    async for event in agent.chat_stream("恢复问题", run_id=run_id, resume=True):
        events.append(event)

    loaded = agent.session_store.load_session(agent.user_id, agent.session_id)
    user_messages = [m for m in loaded.messages if m["role"] == "user"]

    assert len(user_messages) == 1
    assert loaded.messages[-1]["content"] == "恢复后的最终回答"
    assert any("恢复点" in event.get("message", "") for event in events)


@pytest.mark.asyncio
async def test_agent_resume_without_completed_tool_checkpoint_retries_tool(agent_factory):
    """验证工具执行中崩溃且无完成 checkpoint 时，恢复会重新执行该工具。"""
    tool_call = json.dumps(
        {"tool": "lookup", "arguments": {"query": "恢复后重试"}},
        ensure_ascii=False,
    )
    agent = agent_factory([tool_call, "重试工具后的最终回答"], {"lookup": "重试结果"})

    events = []
    async for event in agent.chat_stream("恢复问题", run_id="run-retry-tool", resume=True):
        events.append(event)

    assert agent.mcp_client.calls == [("lookup", {"query": "恢复后重试"})]
    assert any(event["type"] == "tool_result" and event["status"] == "success" for event in events)
    assert "".join(event.get("content", "") for event in events if event["type"] == "content") == "重试工具后的最终回答"


@pytest.mark.asyncio
async def test_agent_running_stream_can_be_cancelled_without_final_done(agent_factory, fake_runtime):
    """验证运行中任务被取消后不会继续输出最终完成事件。"""
    run_id = "run-cancel-stream"
    agent = agent_factory(["这是一个会被取消的较长回答" * 4])
    events = []
    content_count = 0

    async for event in agent.chat_stream("请生成长回答", run_id=run_id):
        events.append(event)
        if event["type"] == "content":
            content_count += 1
        if content_count == 2:
            fake_runtime.request_run_cancel(run_id)

    assert fake_runtime.runs[run_id]["status"] == "cancelled"
    assert not any(event["type"] == "done" for event in events)


@pytest.mark.asyncio
async def test_start_new_session_clears_without_forced_archive(agent_factory):
    """验证开启新会话只清空当前消息，不会强制归档未归档历史。"""
    agent = agent_factory(["不应调用"])
    agent.session.add_message("user", "这段未归档历史不应被强制归档")
    agent.session.add_message("assistant", "保持普通会话保存语义")
    agent.session_store.save_session(agent.session)

    result = await agent.start_new_session()
    loaded = agent.session_store.load_session(agent.user_id, agent.session_id)

    assert result["archived"] == 0
    assert "未强制归档" in result["message"]
    assert loaded.messages == []
    assert agent.session_store.read_memory_md(agent.user_id) == ""
    assert "未归档历史" not in agent.session_store.read_history_md(agent.user_id)
    assert len(agent.llm_client.requests) == 0
