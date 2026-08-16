"""集成测试：覆盖 TravelAgent 内部组件协作和异常分支。"""

import asyncio
import json

import pytest
from langgraph.graph import END
from langgraph.types import Command

from tests.conftest import FakeLLMClient, FakeMCPClient
from travel_agent import TravelAgent


pytestmark = [pytest.mark.integration, pytest.mark.full]


@pytest.mark.asyncio
async def test_agent_context_events_active_run_and_prompt_helpers(agent_factory, fake_runtime):
    """验证 Agent 上下文、事件记录、活跃 run 表和 prompt helper 的协作。"""
    agent = agent_factory(["最终回答"])
    agent.system_prompt = "system prompt"
    agent.session.add_message("user", "历史问题")
    agent.session_store.save_session(agent.session)

    class VectorStore:
        def search_history(self, user_input, top_k=2):
            return [{"full_text": f"{user_input} 的历史", "score": 0.9}]

    agent.vector_store = VectorStore()
    queue = asyncio.Queue()

    assert "相关历史记录" in agent._build_relevant_history_context("上海")
    assert agent._get_context_messages()[0]["content"] == "历史问题"
    assert "上下文 token" in agent._build_context_token_message()
    assert agent._tool_display_name("task") == "子 Agent"
    assert agent._tool_display_name("ask_clarification") == "澄清确认"
    assert agent._summarize_tool_result({"secret": "x"}, limit=10).endswith("...(truncated)")

    await agent._emit_stream_event(queue, "status", "run-event", "事件消息", status="running", tool="lookup")
    event = await queue.get()

    assert event["message"] == "事件消息"
    assert fake_runtime.steps[-1]["tool_name"] == "lookup"

    task = asyncio.create_task(asyncio.sleep(10))
    TravelAgent._mark_run_active("active-run", task, "u", "s")
    try:
        assert TravelAgent.is_run_active("active-run")
        assert TravelAgent.cancel_session_run("u", "s") == ("active-run", True)
    finally:
        task.cancel()
        TravelAgent._mark_run_inactive("active-run")

    assert not TravelAgent.is_run_active("active-run")


@pytest.mark.asyncio
async def test_agent_llm_tool_nodes_and_resumable_workflow_branches(agent_factory, fake_runtime):
    """验证 LLM 节点、工具节点、Command 合并和恢复工作流最大轮次分支。"""
    tool_call = json.dumps({"tool": "lookup", "arguments": {"q": "上海"}}, ensure_ascii=False)
    agent = agent_factory([tool_call, "最终回答"], {"lookup": "工具结果"})
    agent.system_prompt = agent._build_system_prompt()
    state = {
        "messages": [{"role": "user", "content": "查一下"}],
        "tool_calls": [],
        "final_response": "",
        "request_id": "run-node",
        "retrieval_context": "## 相关历史记录\n1. 历史",
        "event_queue": asyncio.Queue(),
        "checkpoint_enabled": True,
    }

    llm_state = await agent.llm_node(state)
    assert llm_state["tool_calls"][0]["tool"] == "lookup"
    tool_state = await agent.tool_node(llm_state)

    assert "工具执行完成" in tool_state["messages"][-1]["content"]
    assert fake_runtime.checkpoints["run-node"][-1]["next_node"] == "llm"

    command = Command(
        update={
            "messages": [{"role": "assistant", "content": "clarified"}],
            "final_response": "clarified",
            "retrieval_context": "ctx",
        },
        goto=END,
    )
    merged = agent._merge_command_update(dict(tool_state), command)

    assert merged["messages"][-1]["content"] == "clarified"
    assert agent._next_node_from_command(merged, command) == "end"
    assert agent.create_graph() is agent.create_graph()

    async def no_progress_llm(state):
        return {**state, "tool_calls": [], "final_response": ""}

    agent.llm_node = no_progress_llm  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="最大迭代次数"):
        await agent._run_resumable_workflow(
            {
                **state,
                "messages": [{"role": "user", "content": "循环"}],
                "tool_calls": [],
                "final_response": "",
            },
            max_iterations=1,
        )


@pytest.mark.asyncio
async def test_agent_tool_node_clarification_task_and_failure_paths(agent_factory):
    """验证澄清、task 空描述、本地工具失败和 MCP 连接失败的工具节点分支。"""
    agent = agent_factory([])
    agent.system_prompt = agent._build_system_prompt()
    queue = asyncio.Queue()
    clarify_state = {
        "messages": [{"role": "user", "content": "帮我订票"}],
        "tool_calls": [
            {
                "tool": "ask_clarification",
                "arguments": {"question": "哪天出发？", "context": "缺少日期", "options": ["今天"]},
            }
        ],
        "final_response": "",
        "request_id": "run-clarify-direct",
        "retrieval_context": "",
        "event_queue": queue,
        "checkpoint_enabled": True,
    }

    command = await agent.tool_node(clarify_state)
    assert isinstance(command, Command)
    assert "哪天出发" in command.update["final_response"]

    task_state = {
        **clarify_state,
        "tool_calls": [{"tool": "task", "arguments": {"subagent_type": "general-purpose"}}],
        "final_response": "",
        "request_id": "run-task-empty",
    }
    task_result = await agent.tool_node(task_state)
    assert "task_description 参数不能为空" in task_result["messages"][-1]["content"]

    local_fail_state = {
        **clarify_state,
        "tool_calls": [{"tool": "read_file", "arguments": {"path": ""}}],
        "final_response": "",
        "request_id": "run-local-fail",
    }
    local_result = await agent.tool_node(local_fail_state)
    assert "Unknown path" in local_result["messages"][-1]["content"]

    failing_agent = agent_factory([], {"lookup": RuntimeError("重连失败: down")})
    failing_state = {
        **clarify_state,
        "tool_calls": [{"tool": "lookup", "arguments": {"q": "x"}}],
        "final_response": "",
        "request_id": "run-mcp-fail",
    }
    failed = await failing_agent.tool_node(failing_state)

    assert "暂时不可用" in failed["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_agent_sync_chat_persists_consolidates_and_reports_errors(agent_factory):
    """验证同步 chat 会持久化消息、触发整合，并在图执行异常时向上抛出。"""
    agent = agent_factory(["同步回答"])

    class Consolidator:
        def __init__(self):
            self.called = False

        async def maybe_consolidate(self, session, system_prompt, request_id=None):
            self.called = True
            return False

    consolidator = Consolidator()
    agent.consolidator = consolidator

    result = await agent.chat("同步问题")
    loaded = agent.session_store.load_session(agent.user_id, agent.session_id)

    assert result == "同步回答"
    assert loaded.messages[-1]["content"] == "同步回答"
    assert consolidator.called is True

    failing_agent = agent_factory([])

    class FailingGraph:
        async def ainvoke(self, state):
            raise RuntimeError("graph failed")

    failing_agent.create_graph = lambda: FailingGraph()  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="graph failed"):
        await failing_agent.chat("失败问题")


@pytest.mark.asyncio
async def test_agent_stream_memory_consolidation_queue_and_inline_paths(agent_factory, fake_runtime):
    """验证流式任务结束后，记忆归档可进入后台队列或本地同步执行。"""
    class StreamConsolidator:
        context_window_tokens = 100
        max_completion_tokens = 10
        safety_buffer = 5
        budget = 85

        def __init__(self):
            self.inline_called = False

        def should_consolidate(self, session, system_prompt):
            return True

        def estimate_session_tokens(self, session, system_prompt):
            return 12

        async def consolidate(self, session, system_prompt, request_id=None):
            self.inline_called = True
            return True

    queued_agent = agent_factory(["队列回答"])
    queued_consolidator = StreamConsolidator()
    queued_agent.consolidator = queued_consolidator
    fake_runtime.enqueue_result = True

    queued_events = []
    async for event in queued_agent.chat_stream("需要归档", run_id="run-queued-memory"):
        queued_events.append(event)

    inline_agent = agent_factory(["本地回答"])
    inline_consolidator = StreamConsolidator()
    inline_agent.consolidator = inline_consolidator
    fake_runtime.enqueue_result = False

    inline_events = []
    async for event in inline_agent.chat_stream("需要本地归档", run_id="run-inline-memory"):
        inline_events.append(event)

    assert any(event.get("status") == "queued" for event in queued_events)
    assert queued_consolidator.inline_called is False
    assert any("正在归档长期记忆" in event.get("message", "") for event in inline_events)
    assert inline_consolidator.inline_called is True


@pytest.mark.asyncio
async def test_agent_cleanup_closes_owned_resources(agent_factory):
    """验证 Agent cleanup 会容忍关闭异常，并断开自有 MCP 客户端。"""
    agent = agent_factory(["ok"])
    agent._owns_mcp_client = True

    class BadConsolidator:
        async def close(self):
            raise RuntimeError("close consolidator failed")

    class AsyncLLM:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    class MCP:
        def __init__(self):
            self.disconnected = False

        async def disconnect(self):
            self.disconnected = True

    llm = AsyncLLM()
    mcp = MCP()
    agent.consolidator = BadConsolidator()
    agent.llm_client = llm
    agent.mcp_client = mcp

    await agent.cleanup()

    assert llm.closed is True
    assert mcp.disconnected is True
