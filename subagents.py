"""
子代理模块：定义子代理配置、system_prompt、SubAgentRunner
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, TypedDict, Sequence

from langgraph.graph import StateGraph, START, END
from langfuse.openai import AsyncOpenAI

from config import MODEL_CONFIG, CONCURRENCY_CONFIG
from observability import get_langfuse_metadata
from logger import error, info, ok


# ── prompt 文件加载 ───────────────────────────────────────────────────

_PROMPT_DIR = Path(__file__).parent / "prompt"


def _load_prompt(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8")


# ── 子代理 system_prompt ──────────────────────────────────────────────

TASK_TOOL_SYSTEM_PROMPT = """## `task`（子代理调度工具）

你可以使用 `task` 工具启动短生命周期的子代理，处理适合隔离执行的任务。子代理只在当前任务期间存在，结束后只返回一份结果给主代理。

### 适合使用 `task` 的场景

- 任务复杂、步骤较多，而且可以独立完成
- 多个子任务彼此独立，适合并行处理
- 子任务需要集中推理、较多上下文或较重的工具调用，拆出去更稳定
- 你只关心子代理的最终结果，不需要保留它的中间过程

### 不适合使用 `task` 的场景

- 任务很简单，只需直接回答或少量工具调用
- 主线程必须持续保留上下文，拆分后反而更混乱
- 拆分只会增加延迟，不能提升质量或稳定性

### 使用要求

1. 明确写出子代理角色、目标、约束和期望输出
2. 能并行的独立任务尽量并行委派
3. 子代理返回后，主代理负责整合结果并面向用户作答

### 调用格式

{"tool": "task", "arguments": {"subagent_type": "子代理类型名称", "task_description": "详细描述子代理需要完成的任务"}}
"""

TASK_TOOL_DESCRIPTION = (
    "启动一个短生命周期的子代理，处理独立且多步骤的任务，并返回精简结果。"
    "参数: subagent_type (子代理类型), task_description (任务描述)。"
)

GENERAL_PURPOSE_SUBAGENT_PROMPT = _load_prompt("GENERAL_PURPOSE_SUBAGENT_PROMPT.md")

TRAVEL_RESEARCH_SUBAGENT_PROMPT = _load_prompt("TRAVEL_RESEARCH_SUBAGENT_PROMPT.md")


# ── 数据类 ────────────────────────────────────────────────────────────

@dataclass
class SubAgentConfig:
    """子代理配置"""
    name: str
    description: str
    system_prompt: str
    max_iterations: int = 10


# ── 默认子代理配置 ────────────────────────────────────────────────────

DEFAULT_SUBAGENTS = [
    SubAgentConfig(
        name="general-purpose",
        description="通用子代理，适合隔离的复杂任务、多步骤执行和聚焦调查。",
        system_prompt=GENERAL_PURPOSE_SUBAGENT_PROMPT,
        max_iterations=10,
    ),
    SubAgentConfig(
        name="travel-research",
        description="旅行研究子代理，适合旅行信息搜索、行程规划和目的地调研。",
        system_prompt=TRAVEL_RESEARCH_SUBAGENT_PROMPT,
        max_iterations=10,
    ),
]


# ── 子代理状态定义 ────────────────────────────────────────────────────

class SubAgentState(TypedDict):
    messages: Sequence[dict]
    tool_calls: list
    final_response: str


# ── SubAgentRunner ────────────────────────────────────────────────────

class SubAgentRunner:
    """
    子代理运行器：创建独立的 LangGraph StateGraph 实例，
    在隔离上下文中执行子代理任务，返回精炼结果。
    """

    _llm_semaphore: ClassVar[asyncio.Semaphore | None] = None
    _llm_semaphore_limit: ClassVar[int] = 0

    def __init__(self, llm_client: AsyncOpenAI, model: str, mcp_client, tools, subagent_configs: list[SubAgentConfig] = None):
        """
        Args:
            llm_client: 共享的 AsyncOpenAI 客户端
            model: 模型名称
            mcp_client: 共享的 MCPClient 实例
            tools: 共享的本地 ToolRegistry 实例
            subagent_configs: 子代理配置列表，默认使用 DEFAULT_SUBAGENTS
        """
        self.llm_client = llm_client
        self.model = model
        self.mcp_client = mcp_client
        self.tools = tools
        self.subagent_configs = {cfg.name: cfg for cfg in (subagent_configs or DEFAULT_SUBAGENTS)}

    @classmethod
    def _get_llm_semaphore(cls) -> asyncio.Semaphore | None:
        """子代理复用同一个 LLM 并发阀门，避免主 Agent 拆任务后绕过限流。"""
        limit = int(CONCURRENCY_CONFIG.get("llm_limit") or 0)
        if limit <= 0:
            return None
        if cls._llm_semaphore is None or cls._llm_semaphore_limit != limit:
            cls._llm_semaphore = asyncio.Semaphore(limit)
            cls._llm_semaphore_limit = limit
        return cls._llm_semaphore

    async def _create_llm_completion(self, **kwargs):
        """统一包一层子代理 LLM 调用，便于压测时控制外部模型并发。"""
        semaphore = self._get_llm_semaphore()
        if semaphore is None:
            return await self.llm_client.chat.completions.create(**kwargs)
        async with semaphore:
            return await self.llm_client.chat.completions.create(**kwargs)

    def get_available_subagents_desc(self) -> str:
        """获取可用子代理的描述文本（用于主代理系统提示词）"""
        lines = []
        for cfg in self.subagent_configs.values():
            lines.append(f"- `{cfg.name}`: {cfg.description}")
        return "\n".join(lines)

    def get_task_tool_description(self) -> str:
        """获取 task 工具的完整描述（用于主代理工具列表）"""
        available = ", ".join(f"'{name}'" for name in self.subagent_configs.keys())
        return (
            f"{TASK_TOOL_DESCRIPTION} "
            f"可用子代理类型: {available}。"
        )

    async def run(
        self,
        subagent_type: str,
        task_description: str,
        max_iterations: int = None,
        session_id: str | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> str:
        """
        执行子代理任务

        Args:
            subagent_type: 子代理类型名称
            task_description: 任务描述
            max_iterations: 最大迭代轮次（覆盖默认值）

        Returns:
            子代理执行结果字符串
        """
        config = self.subagent_configs.get(subagent_type)
        if config is None:
            available = ", ".join(self.subagent_configs.keys())
            return f"未知的子代理类型 '{subagent_type}'，可用类型: {available}"

        iterations = max_iterations or config.max_iterations

        info("SubAgent", f"{subagent_type} 启动 | task={task_description[:60]}")

        try:
            result = await self._run_subagent_loop(
                config,
                task_description,
                iterations,
                session_id=session_id,
                user_id=user_id,
                request_id=request_id,
            )
            ok("SubAgent", f"{subagent_type} 完成 | result_len={len(result)}")
            return result
        except Exception as e:
            error_msg = f"子代理 '{subagent_type}' 执行失败: {str(e)}"
            error("SubAgent", f"{subagent_type} 错误: {e}")
            return error_msg

    async def _run_subagent_loop(
        self,
        config: SubAgentConfig,
        task_description: str,
        max_iterations: int,
        session_id: str | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> str:
        """
        子代理的核心执行循环：构建独立的 LangGraph StateGraph，执行迭代

        Args:
            config: 子代理配置
            task_description: 任务描述
            max_iterations: 最大迭代轮次

        Returns:
            子代理最终回复
        """
        # 构建子代理的 system_prompt：专属 prompt + 可用工具描述 + 工作指令
        system_prompt = self._build_subagent_system_prompt(config)

        # 初始消息：任务描述作为用户输入
        messages = [{"role": "user", "content": task_description}]

        for iteration in range(max_iterations):
            info("SubAgent", f"{config.name} 迭代 {iteration + 1}/{max_iterations}")

            # 调用 LLM
            llm_messages = [{"role": "system", "content": system_prompt}] + messages
            response = await self._create_llm_completion(
                model=self.model,
                messages=llm_messages,
                temperature=0.1,
                name=f"subagent-{config.name}-llm",
                metadata=get_langfuse_metadata(
                    session_id=session_id,
                    user_id=user_id,
                    request_id=request_id,
                    component=f"subagent:{config.name}",
                    operation=f"iteration-{iteration + 1}",
                ),
            )
            content = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": content})

            # 解析工具调用
            tool_call = self._parse_tool_call(content)

            if tool_call is None:
                # 无工具调用，子代理返回最终结果
                return content

            # 执行工具调用
            tool_name = tool_call["tool"]
            tool_args = tool_call["arguments"]

            # 递归防护：子代理不可调用 task 工具
            if tool_name == "task":
                messages.append({
                    "role": "user",
                    "content": "错误：子代理内部不可再调用 task 工具。请直接使用可用工具完成任务，或返回当前结果。"
                })
                continue

            try:
                result = await self._execute_tool(tool_name, tool_args)
                result_message = (
                    f"[工具执行完成]\n"
                    f"工具名称: {tool_name}\n"
                    f"输入参数: {tool_args}\n"
                    f"执行结果: {result}\n\n"
                    f"请根据以上结果继续执行，如果信息已充分请直接返回最终结果。"
                )
                messages.append({"role": "user", "content": result_message})
            except Exception as e:
                error_message = f"工具'{tool_name}'执行失败: {str(e)}\n请尝试其他方法或返回当前结果。"
                messages.append({"role": "user", "content": error_message})

        # 超过最大迭代轮次，返回最后一条助手消息
        last_assistant = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break

        if last_assistant:
            return f"[子代理达到最大迭代轮次({max_iterations})，以下为当前结果]\n{last_assistant}"
        return f"子代理 '{config.name}' 在 {max_iterations} 轮迭代后未能产生有效结果。"

    def _build_subagent_system_prompt(self, config: SubAgentConfig) -> str:
        """构建子代理的完整系统提示词"""
        parts = []

        # 1. 专属 system_prompt
        parts.append(config.system_prompt)

        # 2. 可用工具描述
        mcp_tools_desc = self.mcp_client.get_tools_description()
        if mcp_tools_desc:
            parts.append(f"## MCP 可用工具\n\n{mcp_tools_desc}")

        local_tools_desc = self._get_local_tools_description()
        if local_tools_desc:
            parts.append(f"## 本地工具\n\n{local_tools_desc}")

        # 3. 工作指令
        parts.append(self._get_subagent_workflow())

        return "\n\n---\n\n".join(parts)

    def _get_local_tools_description(self) -> str:
        """获取本地工具描述（排除 task 工具）"""
        if not self.tools.tool_names:
            return ""

        lines = []
        for tool_name in self.tools.tool_names:
            if tool_name == "task":
                continue  # 子代理不可使用 task
            tool = self.tools.get(tool_name)
            if tool:
                lines.append(f"- **{tool_name}**: {tool.description}")
        return "\n".join(lines)

    def _get_subagent_workflow(self) -> str:
        """获取子代理的工作流程指令"""
        return """## 工具调用格式
当需要使用工具时，你的回复必须只包含JSON对象，格式如下:
{"tool": "工具名称", "arguments": {"参数名": "参数值"}}

## 工作流程
1. 理解任务目标，判断是否需要工具
2. 需要工具时返回纯JSON（无其他文字）
3. 收到工具结果后:
   - 需要更多信息 → 调用其他工具（新的JSON）
   - 信息充分 → 直接返回最终结果
4. **不要重复调用相同工具和参数**
5. **不可调用 task 工具**

## 输出要求
- 调用工具: 只返回JSON
- 最终结果: 精炼、结构化、信息密度高"""

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """
        执行工具调用：优先本地工具，否则 MCP 工具
        """
        if self.tools.has(tool_name):
            info("SubAgent", f"调用本地工具 {tool_name}")
            result = await self.tools.execute(tool_name, tool_args)
            return result
        else:
            info("SubAgent", f"调用 MCP 工具 {tool_name}")
            result = await self.mcp_client.call_tool(tool_name, tool_args)
            return result

    @staticmethod
    def _parse_tool_call(content: str) -> dict:
        """
        解析工具调用（与 TravelAgent._parse_tool_call 相同逻辑）
        """
        content = content.strip()

        # 情况1: 直接是 JSON
        try:
            data = json.loads(content)
            if "tool" in data and "arguments" in data:
                return data
        except:
            pass

        # 情况2: Markdown 代码块
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if "tool" in data and "arguments" in data:
                    return data
            except:
                pass

        # 情况3: 直接查找 JSON 对象
        match = re.search(r'\{[^{}]*"tool"[^{}]*"arguments"[^{}]*\}', content)
        if match:
            try:
                data = json.loads(match.group(0))
                if "tool" in data and "arguments" in data:
                    return data
            except:
                pass

        # 情况4: 查找最外层的大括号
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and start < end:
            try:
                data = json.loads(content[start:end + 1])
                if "tool" in data and "arguments" in data:
                    return data
            except:
                pass

        return None
