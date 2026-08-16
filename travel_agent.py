"""
定义了 Agent 的核心逻辑：LLM 推理、工具调用、记忆管理、子代理、支持流式输出
"""

import json
import inspect
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, TypedDict, Sequence
from observability import (
    flush_langfuse,
    get_langfuse_metadata,
    init_langfuse,
    langfuse_request_trace,
    langfuse_tool_span,
    update_langfuse_observation,
)

init_langfuse()
from langfuse.openai import AsyncOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from mcp_client import MCPClient
from config import (
    MODEL_CONFIG,
    SUBAGENT_CONFIG,
    CLARIFICATION_CONFIG,
    AGENT_CONFIG,
    CONCURRENCY_CONFIG,
    LOCAL_TOOL_CONFIG,
)
from industrial_runtime import get_industrial_runtime, sanitize_payload, safe_json
from session_store import SessionStore, SessionData
from memory_manager import VectorHistoryStore
from memory_consolidator import MemoryConsolidator
from skills_loader import SkillsLoader
from tools.registry import ToolRegistry
from tools.shell import ExecTool
from logger import error, info, ok, section, warn
from tools.filesystem import ReadFileTool
from tools.clarification import AskClarificationTool
from tools.web import WebFetchTool, WebSearchTool
from subagents import SubAgentRunner, SubAgentConfig, DEFAULT_SUBAGENTS, TASK_TOOL_SYSTEM_PROMPT
from tool_guard import ToolCallGuard
import asyncio
import threading
import uuid

# prompt 文件目录
_PROMPT_DIR = Path(__file__).parent / "prompt"

# 定义状态
class AgentState(TypedDict):
    messages: Sequence[dict]  # 对话历史
    tool_calls: list  # 当前需要调用的工具
    tool_call_chain: list  # 本轮已执行的完整工具调用链
    final_response: str
    request_id: str
    retrieval_context: str
    event_queue: Any


class TravelAgent:
    """Travel Agent - 支持工具调用和长期记忆"""

    _active_run_ids: ClassVar[set[str]] = set()
    _active_run_ids_lock: ClassVar[threading.Lock] = threading.Lock()
    _active_run_tasks: ClassVar[dict[str, asyncio.Task]] = {}
    _active_run_sessions: ClassVar[dict[str, tuple[str, str]]] = {}
    _active_run_tasks_lock: ClassVar[threading.Lock] = threading.Lock()
    _llm_semaphore: ClassVar[asyncio.Semaphore | None] = None
    _llm_semaphore_limit: ClassVar[int] = 0

    def __init__(self, enable_memory: bool = True, session_id: str = None, user_id: str = None, skills_info: str = "",
                 context_window_tokens: int = 9000, max_completion_tokens: int = 1024,
                 workspace: str = None, mcp_client: MCPClient | None = None):
        self.mcp_client = mcp_client or MCPClient()
        self._owns_mcp_client = mcp_client is None
        self.llm_client = AsyncOpenAI(
            api_key=MODEL_CONFIG["api_key"],
            base_url=MODEL_CONFIG["base_url"]
        )
        self.model = MODEL_CONFIG["model_name"]
        self.system_prompt = ""
        self._graph = None
        self.max_workflow_iterations = AGENT_CONFIG["max_iterations"]
        self.tool_guard = ToolCallGuard()
        self._static_prompt_cache_key = None
        self._static_prompt_before_memory = ""
        self._static_prompt_after_memory = ""
        self.session_id = session_id
        self.user_id = user_id or session_id
        self.runtime = get_industrial_runtime()

        # 初始化 SkillsLoader
        self.skills_loader = SkillsLoader()
        self.skills_info = skills_info  # 外部传入的 skill 元数据（优先级更高）

        # 上下文窗口配置
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens

        # 初始化记忆系统
        self.enable_memory = enable_memory
        self.session_store = None
        self.vector_store = None
        self.consolidator = None
        self.session = None  # SessionData 对象

        # 初始化本地工具注册表（用于 skill 执行）
        self.tools = ToolRegistry()
        self._register_default_tools(workspace)

        # 初始化子代理运行器
        self.subagent_runner = None
        if SUBAGENT_CONFIG.get("enabled", True):
            subagent_configs = [
                SubAgentConfig(
                    name=cfg["name"],
                    description=cfg["description"],
                    system_prompt=cfg["system_prompt"],
                    max_iterations=cfg.get("max_iterations", 8),
                )
                for cfg in SUBAGENT_CONFIG.get("subagents", [])
            ] if SUBAGENT_CONFIG.get("subagents") else DEFAULT_SUBAGENTS

            self.subagent_runner = SubAgentRunner(
                llm_client=self.llm_client,
                model=self.model,
                mcp_client=self.mcp_client,
                tools=self.tools,
                subagent_configs=subagent_configs,
            )
            info("Agent", f"子代理已启用 | types={list(self.subagent_runner.subagent_configs.keys())}")

        if enable_memory and self.user_id and session_id:
            try:
                # 初始化会话存储
                self.session_store = SessionStore()
                # 加载或创建会话
                self.session = self.session_store.load_session(self.user_id, session_id)

                # 初始化向量历史存储
                self.vector_store = VectorHistoryStore(self.user_id)
                # 加载并索引现有 HISTORY.md
                history_content = self.session_store.read_history_md(self.user_id)
                if history_content.strip() != "# Conversation History\n\n":
                    self.vector_store.index_history(history_content)

                # 初始化记忆整合器（预算驱动）
                self.consolidator = MemoryConsolidator(
                    session_store=self.session_store,
                    vector_store=self.vector_store,
                    user_id=self.user_id,
                    context_window_tokens=context_window_tokens,
                    max_completion_tokens=max_completion_tokens,
                    safety_buffer=1024
                )

                ok("Agent", f"记忆系统初始化完成 | uid={self.user_id[:8]} sid={session_id[:8]}")
            except Exception as e:
                warn("Agent", f"记忆系统初始化失败: {e}")
                self.enable_memory = False

    def _register_default_tools(self, workspace: str = None) -> None:
        """注册默认的工具集，用于技能执行。"""
        # 设置工作目录
        if workspace is None:
            workspace = os.getcwd()
        workspace_path = Path(workspace)

        # 文件读取仅允许项目目录和 skills 目录，避免公网请求读取主机敏感文件。
        skills_dir = self.skills_loader.skills_dir if hasattr(self.skills_loader, 'skills_dir') else None
        extra_dirs = [skills_dir] if skills_dir else None
        if LOCAL_TOOL_CONFIG.get("read_file_enabled", True):
            self.tools.register(ReadFileTool(
                workspace=workspace_path,
                allowed_dir=workspace_path,
                extra_allowed_dirs=extra_dirs,
            ))

        # 内置只读联网工具。无需外部 MCP 或额外 API Key；所有 URL 读取均经过
        # 公网地址校验、端口限制、超时和响应大小限制。
        if LOCAL_TOOL_CONFIG.get("web_enabled", True):
            self.tools.register(WebSearchTool(
                endpoint=LOCAL_TOOL_CONFIG.get("web_search_endpoint"),
                timeout=LOCAL_TOOL_CONFIG.get("web_timeout_seconds", 15),
            ))
            self.tools.register(WebFetchTool(
                timeout=LOCAL_TOOL_CONFIG.get("web_timeout_seconds", 15),
            ))

        # 公网部署默认不注册 Shell；只有显式开启时才启用沙箱执行。
        if LOCAL_TOOL_CONFIG.get("shell_enabled", True):
            self.tools.register(ExecTool(
                working_dir=str(workspace_path),
                use_sandbox=True,
                sandbox_config=None  # 自动搜索 .srt-settings.json
            ))

        # 注册 ask_clarification 工具（缺信息时主动向用户澄清）
        if CLARIFICATION_CONFIG.get("enabled", True):
            self.tools.register(AskClarificationTool())

        info("Agent", f"已注册本地工具: {self.tools.tool_names}")

    @classmethod
    def _get_llm_semaphore(cls) -> asyncio.Semaphore | None:
        """按进程共享 LLM 并发阀门，避免高并发评测时把模型服务打爆。"""
        limit = int(CONCURRENCY_CONFIG.get("llm_limit") or 0)
        if limit <= 0:
            return None
        if cls._llm_semaphore is None or cls._llm_semaphore_limit != limit:
            cls._llm_semaphore = asyncio.Semaphore(limit)
            cls._llm_semaphore_limit = limit
        return cls._llm_semaphore

    async def _create_llm_completion(self, **kwargs):
        """统一包一层 LLM 调用，方便按环境变量控制全局并发。"""
        semaphore = self._get_llm_semaphore()
        if semaphore is None:
            return await self.llm_client.chat.completions.create(**kwargs)
        async with semaphore:
            return await self.llm_client.chat.completions.create(**kwargs)

    def _build_system_prompt_uncached(self) -> str:
        """构建系统提示词（参考 nanobot 的 ContextBuilder）"""
        # 获取 MCP 工具描述
        mcp_tools_desc = self.mcp_client.get_tools_description()

        # 获取本地工具描述（用于 skill 执行）
        local_tools_desc = self._get_local_tools_description()

        # 获取 MEMORY.md 内容
        memory_content = ""
        if self.session_store and self.user_id:
            memory_content = self.session_store.read_memory_md(self.user_id)

        # 构建提示词（按照 nanobot 的顺序）
        parts = []

        # 1. 主 agent 提示词（身份 + 工作流程）
        parts.append(self._get_base_system_prompt())

        # 2. MCP 可用工具
        parts.append(f"## MCP 可用工具\n\n{mcp_tools_desc}")

        # 3. 本地工具
        if local_tools_desc:
            parts.append(f"## 本地工具\n\n{local_tools_desc}")

        # 4. task 子代理工具
        if self.subagent_runner:
            task_desc = self.subagent_runner.get_task_tool_description()
            parts.append(f"## 子代理工具\n\n- **task**: {task_desc}")

        # 5. Skill 信息（渐进式加载）
        skills_section = self._build_skills_section()
        if skills_section:
            parts.append(skills_section)

        # 6. 长期记忆（仅当内容非空时）
        if memory_content and memory_content.strip():
            parts.append(f"## 长期记忆\n\n{memory_content}")

        # 7. 子代理使用指南
        if self.subagent_runner:
            subagent_guide = f"{TASK_TOOL_SYSTEM_PROMPT}\n可用子代理类型：\n{self.subagent_runner.get_available_subagents_desc()}"
            parts.append(subagent_guide)

        return "\n\n---\n\n".join(parts)

    def _get_static_system_prompt_parts(self) -> tuple[str, str]:
        cache_key = (
            getattr(self.mcp_client, "tools_version", 0),
            self.skills_info,
            tuple(self.tools.tool_names),
            bool(self.subagent_runner),
        )
        if cache_key == self._static_prompt_cache_key:
            return self._static_prompt_before_memory, self._static_prompt_after_memory

        before_memory = []
        after_memory = []

        mcp_tools_desc = self.mcp_client.get_tools_description()
        local_tools_desc = self._get_local_tools_description()

        before_memory.append(self._get_base_system_prompt())
        before_memory.append(f"## MCP 可用工具\n\n{mcp_tools_desc}")

        if local_tools_desc:
            before_memory.append(f"## 本地工具\n\n{local_tools_desc}")

        if self.subagent_runner:
            task_desc = self.subagent_runner.get_task_tool_description()
            before_memory.append(f"## 子代理工具\n\n- **task**: {task_desc}")

        skills_section = self._build_skills_section()
        if skills_section:
            before_memory.append(skills_section)

        if self.subagent_runner:
            subagent_guide = f"{TASK_TOOL_SYSTEM_PROMPT}\n可用子代理类型：\n{self.subagent_runner.get_available_subagents_desc()}"
            after_memory.append(subagent_guide)

        self._static_prompt_cache_key = cache_key
        self._static_prompt_before_memory = "\n\n---\n\n".join(before_memory)
        self._static_prompt_after_memory = "\n\n---\n\n".join(after_memory)
        return self._static_prompt_before_memory, self._static_prompt_after_memory

    def _build_system_prompt(self) -> str:
        before_memory, after_memory = self._get_static_system_prompt_parts()

        parts = [before_memory]
        memory_content = ""
        if self.session_store and self.user_id:
            memory_content = self.session_store.read_memory_md(self.user_id)

        if memory_content and memory_content.strip():
            parts.append(f"## 长期记忆\n\n{memory_content}")

        if after_memory:
            parts.append(after_memory)

        return "\n\n---\n\n".join(parts)

    def _get_local_tools_description(self) -> str:
        """获取本地工具的描述信息（包含完整参数）"""
        if not self.tools.tool_names:
            return ""

        descriptions = []
        for tool_name in self.tools.tool_names:
            tool = self.tools.get(tool_name)
            if tool:
                desc = f"工具名: {tool_name}\n描述: {tool.description}"
                if tool.parameters and "properties" in tool.parameters:
                    params_desc = []
                    for param_name, param_info in tool.parameters["properties"].items():
                        required = param_name in tool.parameters.get("required", [])
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "无描述")
                        params_desc.append(
                            f"  - {param_name} ({param_type}): {param_desc}{' [必需]' if required else ''}"
                        )
                    if params_desc:
                        desc += f"\n参数:\n{chr(10).join(params_desc)}"
                descriptions.append(desc)
        return "\n\n".join(descriptions)

    _base_system_prompt_cache: ClassVar[str | None] = None

    def _get_base_system_prompt(self) -> str:
        """获取主 agent 完整提示词（身份 + 工作流程，来自 prompt/TRAVEL_AGENT_PROMPT.md）"""
        if TravelAgent._base_system_prompt_cache is None:
            TravelAgent._base_system_prompt_cache = (
                (_PROMPT_DIR / "TRAVEL_AGENT_PROMPT.md")
                .read_text(encoding="utf-8")
                .strip()
            )
        return TravelAgent._base_system_prompt_cache

    def _format_clarification(self, payload: dict, fallback_args: dict) -> str:
        """格式化澄清消息为易读文本。"""
        merged = {
            "question": payload.get("question") or fallback_args.get("question") or "请补充更多信息。",
            "clarification_type": payload.get("clarification_type") or fallback_args.get("clarification_type") or "missing_info",
            "context": payload.get("context") or fallback_args.get("context") or "",
            "options": payload.get("options") or fallback_args.get("options") or [],
        }
        type_icons = {
            "missing_info": "❓",
            "ambiguous_requirement": "❓",
            "approach_choice": "👉",
            "risk_confirmation": "⚠️",
            "suggestion": "💡",
        }
        icon = type_icons.get(merged["clarification_type"], "❓")
        parts: list[str] = []
        if merged["context"]:
            parts.append(f"{icon} {merged['context']}")
            parts.append("")
        parts.append(f"{icon} {merged['question']}")
        if merged["options"]:
            parts.append("")
            for index, option in enumerate(merged["options"], 1):
                parts.append(f"  {index}. {option}")
        return "\n".join(parts)

    def _build_skills_section(self) -> str:
        """构建 Skill 信息部分（参考 nanobot 的 ContextBuilder，渐进式加载）"""
        parts = []

        # 如果外部传入了 skills_info，优先使用
        if self.skills_info:
            parts.append("## 技能信息")
            parts.append(self.skills_info)
            return "\n".join(parts)

        # 否则使用 SkillsLoader 加载
        # 1. 获取所有 skills 的元数据（用于打印）
        all_skills = self.skills_loader.list_skills(filter_unavailable=False)
        if all_skills:
            info("Skills", f"发现 {len(all_skills)} 个 skill")
            for skill in all_skills:
                skill_meta = self.skills_loader._get_skill_meta(skill["name"])
                available = self.skills_loader._check_requirements(skill_meta)
                status = "✓" if available else "✗"
                desc = self.skills_loader._get_skill_description(skill["name"])
                info("Skills", f"  {status} {skill['name']}: {desc[:50]}")

        # 2. 获取 always=true 的 skills 并加载完整内容
        always_skills = self.skills_loader.get_always_skills()
        if always_skills:
            info("Skills", f"自动加载 always=true: {always_skills}")
            always_content = self.skills_loader.load_skills_for_context(always_skills)
            if always_content:
                parts.append("# 激活的技能\n\n这些技能已加载到上下文中，你可以直接使用它们：")
                parts.append(always_content)

        # 3. 获取所有 skills 的元数据摘要
        skills_summary = self.skills_loader.build_skills_summary()
        if skills_summary:
            parts.append("# 可用技能\n")
            parts.append("如需使用某个技能，请使用 read_file 工具读取其 SKILL.md 文件。")
            parts.append("Skills with available=\"false\" 需要先安装依赖。")
            parts.append("")
            parts.append(skills_summary)

        return "\n\n".join(parts) if parts else ""

    def _build_relevant_history_context(self, user_input: str, top_k: int = 2) -> str:
        """构建本轮临时检索上下文；只给 LLM 参考，不写入 session.jsonl。"""
        if not self.vector_store:
            return ""

        results = self.vector_store.search_history(user_input, top_k=top_k)
        if not results:
            return ""

        lines = [
            "## 相关历史记录",
        ]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['full_text']} (相关度: {r['score']:.2f})")

        return "\n".join(lines)

    def _refresh_session_from_store(self, reason: str = "") -> None:
        if not self.session_store or not self.user_id or not self.session_id:
            return

        try:
            old_message_count = len(self.session.messages) if self.session else 0
            old_last_consolidated = (
                self.session.last_consolidated if self.session else 0
            )
            refreshed = self.session_store.load_session(self.user_id, self.session_id)
            self.session = refreshed

            new_message_count = len(refreshed.messages)
            new_last_consolidated = refreshed.last_consolidated
            if (
                new_message_count != old_message_count
                or new_last_consolidated != old_last_consolidated
            ):
                detail = f" | {reason}" if reason else ""
                info(
                    "Memory",
                    "已刷新会话快照"
                    f"{detail} messages={old_message_count}->{new_message_count}"
                    f" last_consolidated={old_last_consolidated}->{new_last_consolidated}",
                )
        except Exception as exc:
            warn("Memory", f"刷新会话快照失败: {exc}")

    def _get_context_messages(self) -> list:
        """获取上下文中使用的消息列表（从 session 中获取未整合的历史消息）"""
        messages = []

        # 添加未整合的历史消息（last_consolidated 之后的消息，包含当前用户输入）
        if self.session:
            unconsolidated = self.session.get_unconsolidated_messages()
            for msg in unconsolidated:
                messages.append({
                    "role": msg["role"],
                    "content": msg.get("content", "")
                })

        return messages

    def _tool_display_name(self, tool_name: str) -> str:
        if tool_name == "task":
            return "子 Agent"
        if tool_name == "ask_clarification":
            return "澄清确认"
        return tool_name

    def _summarize_tool_result(self, result: Any, limit: int = 160) -> str:
        text = result if isinstance(result, str) else safe_json(result, limit=limit * 2)
        text = " ".join(str(text).split())
        if len(text) > limit:
            return text[:limit] + "...(truncated)"
        return text

    def _get_mcp_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """读取 MCP 工具注册时提供的 inputSchema，用于调用前硬校验。"""
        tool = getattr(self.mcp_client, "tools", {}).get(tool_name)
        if not tool:
            return None
        schema = getattr(tool, "inputSchema", None)
        return schema if isinstance(schema, dict) else None

    def _prepare_tool_arguments(self, tool_name: str, tool_args: Any) -> tuple[dict[str, Any], str | None]:
        """执行前校验并尽量复用本地工具已有的参数转换逻辑。"""
        if not isinstance(tool_args, dict):
            return {}, f"工具参数校验失败：parameters must be an object, got {type(tool_args).__name__}"

        if tool_name == "task":
            if not str(tool_args.get("task_description") or "").strip():
                return tool_args, "工具参数校验失败：task_description 参数不能为空。"
            errors = self.tool_guard.validate_arguments(tool_name, tool_args, ToolCallGuard.TASK_SCHEMA)
            return tool_args, self._format_tool_validation_error(tool_name, errors) if errors else None

        if self.tools.has(tool_name):
            _tool, cast_params, error_text = self.tools.prepare_call(tool_name, tool_args)
            if error_text:
                return cast_params, f"工具参数校验失败：{error_text}"
            return cast_params, None

        errors = self.tool_guard.validate_arguments(
            tool_name,
            tool_args,
            self._get_mcp_tool_schema(tool_name),
        )
        return tool_args, self._format_tool_validation_error(tool_name, errors) if errors else None

    @staticmethod
    def _format_tool_validation_error(tool_name: str, errors: list[str]) -> str | None:
        """把 schema 校验错误整理成 LLM 可修正的提示。"""
        if not errors:
            return None
        return (
            f"工具参数校验失败：工具 '{tool_name}' 的参数不符合 schema："
            + "; ".join(errors)
            + "。请根据用户上下文补齐或修正参数后重新调用工具。"
        )

    async def _call_mcp_tool_with_retry(self, tool_name: str, tool_args: dict[str, Any], tool_call_record: dict[str, Any]) -> Any:
        """对 MCP 调用增加外层分类重试，参数类错误不会盲目重试。"""
        attempt = 1
        max_attempts = 1
        while True:
            try:
                tool_call_record["attempt"] = attempt
                tool_call_record["max_attempts"] = max_attempts
                return await self.mcp_client.call_tool(tool_name, tool_args)
            except Exception as exc:
                retry_decision = self.tool_guard.classify_exception(exc)
                max_attempts = retry_decision.max_attempts
                tool_call_record.update({
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error_type": retry_decision.error_type,
                    "retryable": retry_decision.retryable,
                })
                if not retry_decision.retryable or attempt >= max_attempts:
                    raise
                delay = 0.5 * (3 ** (attempt - 1))
                warn(
                    "Tool",
                    f"MCP 工具 {tool_name} 调用失败，{delay:.1f}s 后重试 "
                    f"attempt={attempt}/{max_attempts} type={retry_decision.error_type} err={exc}",
                )
                await asyncio.sleep(delay)
                attempt += 1

    def _langfuse_completion_metadata(self, request_id: str, mode: str, result: dict[str, Any]) -> dict[str, Any]:
        tool_call_chain = list(result.get("tool_call_chain") or [])
        tool_calls = [
            str(item.get("tool") or item.get("name"))
            for item in tool_call_chain
            if isinstance(item, dict) and (item.get("tool") or item.get("name"))
        ]
        return get_langfuse_metadata(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            component="travel_agent",
            operation="chat",
            extra={
                "mode": mode,
                "tool_calls": tool_calls,
                "tool_call_chain": tool_call_chain,
                "tool_call_count": len(tool_calls),
            },
        )

    async def _emit_stream_event(
        self,
        event_queue: asyncio.Queue | None,
        event_type: str,
        run_id: str,
        message: str = "",
        **payload: Any,
    ) -> None:
        record_step = bool(payload.pop("record_step", True))
        event = {
            "type": event_type,
            "run_id": run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            **payload,
        }
        if message:
            event["message"] = message

        if event_type != "content" and record_step:
            status = str(payload.get("status") or "running")
            self.runtime.record_run_step(
                run_id=run_id,
                step_type=event_type,
                status=status,
                message=message,
                tool_name=payload.get("tool"),
                payload=sanitize_payload(payload),
                elapsed_ms=payload.get("elapsed_ms"),
            )

        if event_queue is not None:
            await event_queue.put(event)

    async def _emit_state_event(
        self,
        state: AgentState,
        event_type: str,
        message: str = "",
        **payload: Any,
    ) -> None:
        # 轻量评测路径不需要写 run steps，避免高并发评测把 MySQL 写入放大。
        payload.setdefault("record_step", bool(state.get("record_steps", True)))
        await self._emit_stream_event(
            state.get("event_queue"),
            event_type,
            state.get("request_id", ""),
            message,
            **payload,
        )

    def get_context_token_usage(self) -> tuple[int, int] | None:
        if not self.session:
            return None

        if self.consolidator:
            estimated = self.consolidator.estimate_session_tokens(
                self.session,
                self.system_prompt,
            )
            budget = self.consolidator.budget
        else:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self._get_context_messages())
            estimated = sum(len(str(msg.get("content", ""))) for msg in messages) // 3
            budget = self.context_window_tokens - self.max_completion_tokens - 1024
        return estimated, budget

    def _build_context_token_message(self) -> str | None:
        try:
            usage = self.get_context_token_usage()
            if not usage:
                return None
            estimated, budget = usage
            return f"上下文 token: {estimated}/{budget}"
        except Exception as exc:
            warn("Memory", f"上下文 token 统计失败: {exc}")
            return None

    @classmethod
    def is_run_active(cls, run_id: str) -> bool:
        with cls._active_run_ids_lock:
            return run_id in cls._active_run_ids

    @classmethod
    def _mark_run_active(
        cls,
        run_id: str,
        task: asyncio.Task | None = None,
        user_id: str = "",
        session_id: str = "",
    ) -> None:
        with cls._active_run_ids_lock:
            cls._active_run_ids.add(run_id)
        if task is not None:
            with cls._active_run_tasks_lock:
                cls._active_run_tasks[run_id] = task
                if user_id and session_id:
                    cls._active_run_sessions[run_id] = (user_id, session_id)

    @classmethod
    def _mark_run_inactive(cls, run_id: str) -> None:
        with cls._active_run_ids_lock:
            cls._active_run_ids.discard(run_id)
        with cls._active_run_tasks_lock:
            cls._active_run_tasks.pop(run_id, None)
            cls._active_run_sessions.pop(run_id, None)

    @classmethod
    def cancel_run(cls, run_id: str) -> bool:
        with cls._active_run_tasks_lock:
            task = cls._active_run_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    @classmethod
    def cancel_session_run(cls, user_id: str, session_id: str) -> tuple[str, bool]:
        with cls._active_run_tasks_lock:
            active_items = list(cls._active_run_sessions.items())
        for run_id, (active_user_id, active_session_id) in reversed(active_items):
            if active_user_id == user_id and active_session_id == session_id:
                return run_id, cls.cancel_run(run_id)
        return "", False

    def _raise_if_run_cancelled(self, run_id: str) -> None:
        if self.runtime.is_run_cancelled(run_id):
            raise asyncio.CancelledError("用户主动停止任务")

    def _has_user_message_for_run(self, user_input: str, run_id: str) -> bool:
        if not self.session:
            return False

        for msg in self.session.messages:
            if msg.get("role") == "user" and msg.get("run_id") == run_id:
                return True

        # 兼容崩溃前已经落盘、但旧消息缺 run_id 的情况：只把最后一个未回答用户消息视为本轮输入。
        for msg in reversed(self.session.messages):
            role = msg.get("role")
            if role == "assistant":
                return False
            if role == "user":
                return str(msg.get("content") or "") == user_input
        return False

    def _checkpoint_copy(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return sanitize_payload(value, limit=5000)

    def _checkpoint_payload(self, state: AgentState, next_node: str) -> dict[str, Any]:
        return {
            "version": 1,
            "run_id": state.get("request_id", ""),
            "next_node": next_node,
            "messages": self._checkpoint_copy(list(state.get("messages") or [])),
            "tool_calls": self._checkpoint_copy(list(state.get("tool_calls") or [])),
            "tool_call_chain": self._checkpoint_copy(list(state.get("tool_call_chain") or [])),
            "final_response": str(state.get("final_response") or ""),
            "retrieval_context": str(state.get("retrieval_context") or ""),
        }

    def _save_run_checkpoint(self, state: AgentState, next_node: str) -> None:
        if not state.get("checkpoint_enabled"):
            return
        run_id = str(state.get("request_id") or "")
        if not run_id:
            return
        try:
            self.runtime.record_run_checkpoint(run_id, self._checkpoint_payload(state, next_node))
        except Exception as exc:
            warn("Agent", f"保存任务恢复点失败 | rid={run_id[:8]} err={exc}")

    def _state_from_checkpoint(
        self,
        checkpoint: dict[str, Any] | None,
        event_queue: asyncio.Queue | None,
    ) -> tuple[AgentState | None, str]:
        if not isinstance(checkpoint, dict):
            return None, "llm"
        next_node = str(checkpoint.get("next_node") or "llm")
        if next_node not in {"llm", "tool", "end"}:
            return None, "llm"

        messages = checkpoint.get("messages")
        tool_calls = checkpoint.get("tool_calls")
        if not isinstance(messages, list) or not isinstance(tool_calls, list):
            return None, "llm"

        state: AgentState = {
            "messages": messages,
            "tool_calls": tool_calls,
            "tool_call_chain": list(checkpoint.get("tool_call_chain") or []),
            "final_response": str(checkpoint.get("final_response") or ""),
            "request_id": str(checkpoint.get("run_id") or ""),
            "retrieval_context": str(checkpoint.get("retrieval_context") or ""),
            "event_queue": event_queue,
            "checkpoint_enabled": True,
        }
        if not state["request_id"]:
            return None, "llm"
        return state, next_node

    def _merge_command_update(self, state: AgentState, command: Command) -> AgentState:
        update = getattr(command, "update", None)
        if not isinstance(update, dict):
            return state

        for key, value in update.items():
            if key == "messages":
                messages = list(state.get("messages") or [])
                if isinstance(value, list):
                    messages.extend(value)
                elif value:
                    messages.append(value)
                state["messages"] = messages
            elif key in {"tool_calls", "tool_call_chain", "final_response", "retrieval_context"}:
                state[key] = value
            else:
                state[key] = value
        return state

    def _next_node_from_command(self, state: AgentState, command: Command) -> str:
        goto = getattr(command, "goto", None)
        if goto == END or goto == "__end__":
            return "end"
        if isinstance(goto, str) and goto in {"llm", "tool"}:
            return str(goto)
        if state.get("final_response"):
            return "end"
        return "llm"

    async def _run_resumable_workflow(
        self,
        initial_state: AgentState,
        start_node: str = "llm",
        max_iterations: int | None = None,
    ) -> AgentState:
        state = initial_state
        node = start_node if start_node in {"llm", "tool", "end"} else "llm"
        iteration_limit = max_iterations or self.max_workflow_iterations

        for _ in range(iteration_limit):
            if node == "end":
                self._save_run_checkpoint(state, "end")
                return state

            if node == "llm":
                state = await self.llm_node(state)
                node = self.should_continue(state)
                if node == "end":
                    self._save_run_checkpoint(state, "end")
                    return state
                continue

            if node == "tool":
                tool_result = await self.tool_node(state)
                if hasattr(tool_result, "update") and hasattr(tool_result, "goto"):
                    state = self._merge_command_update(state, tool_result)
                    node = self._next_node_from_command(state, tool_result)
                    self._save_run_checkpoint(state, node)
                    if node == "end":
                        return state
                    continue

                state = tool_result
                node = self._should_continue_after_tool(state)
                if node == "end":
                    self._save_run_checkpoint(state, "end")
                    return state
                continue

        raise RuntimeError(f"Agent 工作流超过最大迭代次数: {iteration_limit}")

    async def llm_node(self, state: AgentState) -> AgentState:
        """
        LLM推理节点
        """
        forced_web_call = self._build_forced_web_search_call(state)
        if forced_web_call:
            await self._emit_state_event(
                state,
                "status",
                "正在联网搜索...",
                status="running",
                tool="web_search",
            )
            state["tool_calls"] = [forced_web_call]
            state["final_response"] = ""
            self._save_run_checkpoint(state, "tool")
            info("Agent", f"显式联网请求，直接路由 web_search | query={forced_web_call['arguments']['query'][:80]}")
            return state

        # 构建完整消息列表
        await self._emit_state_event(state, "status", "正在思考下一步...", status="running")
        system_prompt = self.system_prompt
        retrieval_context = state.get("retrieval_context", "").strip()
        if retrieval_context:
            system_prompt = f"{system_prompt}\n\n---\n\n{retrieval_context}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(state["messages"])

        # 调用 LLM
        response = await self._create_llm_completion(
            model=self.model,
            messages=messages,
            temperature=0.1,
            name="travel-agent-llm",
            metadata=get_langfuse_metadata(
                session_id=self.session_id,
                user_id=self.user_id,
                request_id=state.get("request_id"),
                component="travel_agent",
                operation="llm_node",
            ),
        )
        content = response.choices[0].message.content.strip()
        state["messages"].append({"role": "assistant", "content": content})

        # 解析工具调用
        tool_call = self._parse_tool_call(content)

        info("LLM", f"响应: {content[:120]}")
        info("LLM", f"解析工具调用: {tool_call}")

        if tool_call:
            await self._emit_state_event(
                state,
                "status",
                f"准备调用 {self._tool_display_name(tool_call.get('tool', 'unknown'))} 工具...",
                status="running",
                tool=tool_call.get("tool"),
            )
            state["tool_calls"] = [tool_call]
            state["final_response"] = ""
        else:
            await self._emit_state_event(state, "status", "正在整理最终回答...", status="running")
            state["final_response"] = content
            state["tool_calls"] = []

        self._save_run_checkpoint(
            state,
            "tool" if state.get("tool_calls") else "end",
        )
        return state

    def _build_forced_web_search_call(self, state: AgentState) -> dict | None:
        """显式联网意图直接进入搜索工具，避免依赖模型自行决定是否调用。"""
        if not self.tools.has("web_search"):
            return None
        if any(
            record.get("tool") == "web_search"
            for record in state.get("tool_call_chain", [])
            if isinstance(record, dict)
        ):
            return None

        user_text = ""
        for message in reversed(state.get("messages", [])):
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "").strip()
            if content and not content.startswith("[工具执行完成]"):
                user_text = content
                break
        if not user_text:
            return None

        if re.search(r"(?:不要|无需|不用|不需要)(?:联网|上网|搜索|检索)", user_text):
            return None
        explicit_web_intent = re.search(
            r"联网|上网|网页搜索|网络搜索|搜索|检索|最新(?:消息|新闻|资讯|动态)|实时(?:信息|数据|新闻)",
            user_text,
            re.IGNORECASE,
        )
        if not explicit_web_intent:
            return None
        return {
            "tool": "web_search",
            "arguments": {"query": user_text[:300], "max_results": 5},
        }

    async def tool_node(self, state: AgentState) -> AgentState | Command:
        """
        工具调用节点
        1. 如果是 ask_clarification → 返回 Command(goto=END) 中断本轮
        2. 如果是 task 工具 → 委派给子代理
        3. 否则如果是本地工具（exec, read_file）→ 直接执行
        4. 否则调用 MCP 工具
        """
        if not state.get("tool_calls"):
            return state

        tool_call = state["tool_calls"][0]
        tool_name = tool_call["tool"]
        tool_args = tool_call["arguments"]
        tool_args, validation_error = self._prepare_tool_arguments(tool_name, tool_args)
        display_name = self._tool_display_name(tool_name)
        sanitized_args = sanitize_payload(tool_args)
        tool_call_chain = state.setdefault("tool_call_chain", [])
        signature = self.tool_guard.signature(tool_name, tool_args)

        if validation_error:
            tool_call_record = {
                "tool": tool_name,
                "arguments": sanitized_args,
                "signature": signature,
                "status": "failed",
                "error_type": "validation",
                "retryable": False,
                "error": validation_error,
            }
            tool_call_chain.append(tool_call_record)
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具参数校验失败",
                status="failed",
                tool=tool_name,
                result_summary=validation_error,
            )
            state["messages"].append({"role": "user", "content": validation_error})
            if self.tool_guard.should_stop_after_failure(tool_call_chain):
                state["final_response"] = self.tool_guard.build_failure_stop_response(tool_call_chain)
            state["tool_calls"] = []
            self._save_run_checkpoint(
                state,
                "end" if state.get("final_response") else "llm",
            )
            return state

        guard_decision = self.tool_guard.before_call(tool_call_chain, tool_name, tool_args)
        if guard_decision.action in {"skip", "terminate"}:
            tool_call_record = {
                "tool": tool_name,
                "arguments": sanitized_args,
                "signature": guard_decision.signature,
                "attempt": guard_decision.attempt,
                "status": "blocked" if guard_decision.action == "terminate" else "skipped",
                "result_summary": guard_decision.last_result_summary or guard_decision.message,
            }
            tool_call_chain.append(tool_call_record)
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具重复调用已拦截",
                status=tool_call_record["status"],
                tool=tool_name,
                result_summary=guard_decision.final_response or guard_decision.message,
            )
            if guard_decision.action == "terminate":
                state["final_response"] = guard_decision.final_response
            else:
                state["messages"].append({"role": "user", "content": guard_decision.message})
            state["tool_calls"] = []
            self._save_run_checkpoint(
                state,
                "end" if state.get("final_response") else "llm",
            )
            return state

        tool_call_record = {
            "tool": tool_name,
            "arguments": sanitized_args,
            "signature": signature,
            "attempt": guard_decision.attempt,
            "status": "running",
        }
        tool_call_chain.append(tool_call_record)
        if tool_name == "ask_clarification":
            started_at = time.perf_counter()
            await self._emit_state_event(
                state,
                "tool_start",
                f"正在调用 {display_name} 工具...",
                status="running",
                tool=tool_name,
                arguments=sanitized_args,
            )

        # 拦截 ask_clarification：返回 Command(goto=END) 中断本轮循环
        if tool_name == "ask_clarification":
            info("Tool", "调用澄清工具 ask_clarification")
            try:
                result_str = await self.tools.execute(tool_name, tool_args)
                payload = json.loads(result_str) if isinstance(result_str, str) else {}
            except Exception as exc:
                payload = {}
                tool_call_record.update({"status": "failed", "error": str(exc)})
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            if tool_call_record.get("status") != "failed":
                tool_call_record.update({
                    "status": "success",
                    "elapsed_ms": elapsed_ms,
                    "result_summary": self._summarize_tool_result(payload),
                })
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具已完成",
                status="success",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=self._summarize_tool_result(payload),
            )
            formatted = self._format_clarification(payload, tool_args)
            checkpoint_state = dict(state)
            checkpoint_state["messages"] = list(state.get("messages") or []) + [
                {"role": "assistant", "content": formatted}
            ]
            checkpoint_state["final_response"] = formatted
            checkpoint_state["tool_calls"] = []
            self._save_run_checkpoint(checkpoint_state, "end")
            return Command(
                update={
                    "messages": [{"role": "assistant", "content": formatted}],
                    "final_response": formatted,
                    "tool_calls": [],
                },
                goto=END,
            )

        started_at = time.perf_counter()
        await self._emit_state_event(
            state,
            "tool_start",
            f"正在调用 {display_name} 工具...",
            status="running",
            tool=tool_name,
            arguments=sanitized_args,
        )

        tool_span_cm = langfuse_tool_span(
            tool_name=tool_name,
            arguments=sanitized_args,
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=state.get("request_id"),
        )
        tool_span = tool_span_cm.__enter__()

        try:
            # 1. 检查是否为 task 子代理工具
            if tool_name == "task" and self.subagent_runner:
                info("Tool", f"调用子代理工具 task | 参数={tool_args}")
                subagent_type = tool_args.get("subagent_type", "general-purpose")
                task_description = tool_args.get("task_description", "")
                if not task_description:
                    result = "错误：task_description 参数不能为空"
                else:
                    result = await self.subagent_runner.run(
                        subagent_type,
                        task_description,
                        session_id=self.session_id,
                        user_id=self.user_id,
                        request_id=state.get("request_id"),
                    )
                info("Tool", f"子代理结果: {result[:120]}")

            # 2. 优先检查本地工具（用于 skill 执行）
            elif self.tools.has(tool_name):
                info("Tool", f"调用本地工具 {tool_name} | 参数={tool_args}")
                result = await self.tools.execute(tool_name, tool_args)
                info("Tool", f"结果: {str(result)[:120]}")
            # 3. 否则调用 MCP 工具
            else:
                info("Tool", f"调用 MCP 工具 {tool_name}")
                result = await self._call_mcp_tool_with_retry(tool_name, tool_args, tool_call_record)

            # 工具以 Error/错误 开头时视为失败，避免把失败结果当作有效新增信息继续循环。
            if isinstance(result, str) and result.lstrip().lower().startswith("error"):
                raise RuntimeError(result)
            if isinstance(result, str) and result.lstrip().startswith("错误"):
                raise RuntimeError(result)

            update_langfuse_observation(
                tool_span,
                output=self._summarize_tool_result(result, limit=500),
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            tool_call_record.update({
                "status": "success",
                "elapsed_ms": elapsed_ms,
                "result_summary": self._summarize_tool_result(result),
            })

            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具已返回",
                status="success",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=self._summarize_tool_result(result),
            )

            result_message = (
                f"[工具执行完成]\n"
                f"工具名称: {tool_name}\n"
                f"输入参数: {tool_args}\n"
                f"执行结果: {result}\n\n"
                f"请根据以上结果：\n"
                f"- 如果需要更多信息，调用其他工具\n"
                f"- 如果信息已充分，用中文向用户总结答案"
            )
            state["messages"].append({"role": "user", "content": result_message})
        except RuntimeError as e:
            update_langfuse_observation(tool_span, level="ERROR", status_message=str(e))
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            tool_call_record.update({
                "status": "failed",
                "elapsed_ms": elapsed_ms,
                "error": str(e),
            })
            if "重连失败" in str(e) or "连接异常" in str(e):
                error_message = (
                    f"工具'{tool_name}'暂时不可用（连接已断开且重连失败）。\n"
                    f"请尝试：\n"
                    f"1. 使用其他可用工具完成任务\n"
                    f"2. 或向用户说明情况并建议稍后重试"
                )
            else:
                error_message = f"工具'{tool_name}'执行失败: {str(e)}\n请尝试其他方法或向用户说明。"
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具调用失败",
                status="failed",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=str(e),
            )
            state["messages"].append({"role": "user", "content": error_message})
            if self.tool_guard.should_stop_after_failure(tool_call_chain):
                state["final_response"] = self.tool_guard.build_failure_stop_response(tool_call_chain)
        except Exception as e:
            update_langfuse_observation(tool_span, level="ERROR", status_message=str(e))
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            tool_call_record.update({
                "status": "failed",
                "elapsed_ms": elapsed_ms,
                "error": str(e),
            })
            error_message = f"工具'{tool_name}'执行失败: {str(e)}\n请尝试其他方法或向用户说明。"
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具调用失败",
                status="failed",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=str(e),
            )
            state["messages"].append({"role": "user", "content": error_message})
            if self.tool_guard.should_stop_after_failure(tool_call_chain):
                state["final_response"] = self.tool_guard.build_failure_stop_response(tool_call_chain)
        finally:
            try:
                tool_span_cm.__exit__(None, None, None)
            except Exception:
                pass

        state["tool_calls"] = []
        self._save_run_checkpoint(state, "end" if state.get("final_response") else "llm")
        return state

    def should_continue(self, state: AgentState) -> str:
        """
        条件判断函数
        """
        if state.get("tool_calls"):
            return "tool"
        elif state.get("final_response"):
            return "end"
        else:
            return "llm"

    def _normalize_tool_call(self, data: Any) -> dict | None:
        """把解析出的 JSON 数据规范化为单个工具调用。"""
        if isinstance(data, dict) and "tool" in data and "arguments" in data:
            if isinstance(data.get("arguments"), dict):
                return data
            return {
                **data,
                "arguments": {},
            }
        if isinstance(data, list):
            for item in data:
                tool_call = self._normalize_tool_call(item)
                if tool_call:
                    return tool_call
        return None

    def _parse_tool_call(self, content: str) -> dict | None:
        """
        解析工具调用
        """
        content = content.strip()
        decoder = json.JSONDecoder()

        # 情况1: 直接是 JSON
        try:
            data = json.loads(content)
            tool_call = self._normalize_tool_call(data)
            if tool_call:
                return tool_call
        except Exception:
            pass

        # 情况2: Markdown 代码块
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                tool_call = self._normalize_tool_call(data)
                if tool_call:
                    return tool_call
            except Exception:
                pass

        # 情况3: 从任意位置扫描 JSON 对象，兼容嵌套 arguments 和一次输出多个 JSON 的情况。
        for match in re.finditer(r"\{", content):
            try:
                data, _ = decoder.raw_decode(content[match.start():])
            except json.JSONDecodeError:
                continue
            tool_call = self._normalize_tool_call(data)
            if tool_call:
                return tool_call

        # 情况4: 查找最外层的大括号
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and start < end:
            try:
                data = json.loads(content[start:end + 1])
                tool_call = self._normalize_tool_call(data)
                if tool_call:
                    return tool_call
            except Exception:
                pass

        return None

    def create_graph(self):
        """
        创建 LangGraph 工作流
        """
        if self._graph is not None:
            return self._graph

        workflow = StateGraph(AgentState)
        workflow.add_node("llm", self.llm_node)
        workflow.add_node("tool", self.tool_node)
        workflow.add_edge(START, "llm")
        workflow.add_conditional_edges(
            "llm",
            self.should_continue,
            {"tool": "tool", "end": END, "llm": "llm"}
        )
        workflow.add_conditional_edges(
            "tool",
            self._should_continue_after_tool,
            {"end": END, "llm": "llm"}
        )

        self._graph = workflow.compile()
        return self._graph

    def _should_continue_after_tool(self, state: AgentState) -> str:
        """
        工具执行后的条件判断
        - 如果 final_response 已设置（如 ask_clarification），则结束
        - 否则继续 LLM 推理
        """
        if state.get("final_response"):
            return "end"
        else:
            return "llm"

    async def chat_stream(self, user_input: str):
        """
        流式处理用户输入（新版）
        """
        request_id = str(uuid.uuid4())[:8]
        info("Agent", f"开始处理 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={request_id}")

        self._refresh_session_from_store("stream request start")

        # 1. 构建系统提示词
        self.system_prompt = self._build_system_prompt()
        token_message = self._build_context_token_message()
        if token_message:
            info("Memory", f"请求开始 {token_message}")

        # 2. 获取相关历史（向量搜索），仅作为本轮临时上下文传给 LLM
        relevant_history = self._build_relevant_history_context(user_input)

        # 3. 添加到会话：只持久化用户原始输入，避免把检索上下文写入 session.jsonl
        if self.session:
            self.session.add_message("user", user_input)

        # 4. 构建上下文消息（从 session 获取未整合的历史，包含当前用户输入）
        context_messages = self._get_context_messages()

        # 打印完整上下文
        section(f"完整上下文 | rid={request_id}")
        info("Context", f"[SYSTEM PROMPT] {self.system_prompt[:200]}")
        info("Context", f"[USER MESSAGES] 共 {len(context_messages)} 条:")
        for i, msg in enumerate(context_messages):
            info("Context", f"  {i+1}. [{msg['role']}] {str(msg['content'])[:140]}")
        section("END 上下文")

        # 5. 执行工作流
        graph = self.create_graph()
        initial_state = {
            "messages": context_messages,
            "tool_calls": [],
            "tool_call_chain": [],
            "final_response": "",
            "request_id": request_id,
            "retrieval_context": relevant_history,
        }

        with langfuse_request_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            input_data=user_input,
            metadata={"mode": "stream"},
        ) as observation:
            try:
                info("Agent", f"执行工作流 | rid={request_id}")
                result = await graph.ainvoke(initial_state)
                ok("Agent", f"工作流完成 | rid={request_id}")

                final_response = result["final_response"]
                update_langfuse_observation(
                    observation,
                    output=final_response,
                    metadata=self._langfuse_completion_metadata(request_id, "stream", result),
                )

                # 7. 添加助手回复到会话
                if self.session and final_response:
                    self.session.add_message("assistant", final_response)

                # 8. 保存会话到 JSONL
                if self.session_store and self.session:
                    self.session_store.save_session(self.session)
                    ok("Agent", f"已保存会话 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={request_id}")

                # 9. 流式输出
                if final_response:
                    for char in final_response:
                        yield char
                        await asyncio.sleep(0.02)

                # 10. 检查并执行记忆整合
                if self.consolidator and self.session:
                    await self.consolidator.maybe_consolidate(
                        self.session,
                        self.system_prompt,
                        request_id=request_id,
                    )
            except Exception as e:
                update_langfuse_observation(
                    observation,
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            finally:
                flush_langfuse()

        ok("Agent", f"完成 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={request_id}")

    async def chat_stream(
        self,
        user_input: str,
        run_id: str | None = None,
        resume: bool = False,
        content_delay: float | None = None,
    ):
        """流式输出当前用户请求的结构化运行事件。"""
        run_id = run_id or uuid.uuid4().hex
        # 非流式 /chat 可传 0 跳过逐字等待，前端流式默认保留打字机节奏。
        output_delay = (
            float(CONCURRENCY_CONFIG["stream_content_delay"])
            if content_delay is None
            else max(0.0, float(content_delay))
        )
        event_queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()
        final_response_holder = {"text": ""}
        partial_response_holder = {"text": ""}

        async def run_workflow() -> None:
            info("Agent", f"开始处理 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={run_id[:8]}")
            self.runtime.record_run_start(
                run_id=run_id,
                user_id=self.user_id or "",
                session_id=self.session_id or "",
                input_text=user_input,
            )

            with langfuse_request_trace(
                session_id=self.session_id,
                user_id=self.user_id,
                request_id=run_id,
                input_data=user_input,
                metadata={"mode": "stream"},
            ) as observation:
                try:
                    self._raise_if_run_cancelled(run_id)
                    if resume:
                        await self._emit_stream_event(
                            event_queue,
                            "status",
                            run_id,
                            "服务重启后正在接管任务...",
                            status="running",
                        )

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在构建上下文...",
                        status="running",
                    )
                    self._raise_if_run_cancelled(run_id)
                    self._refresh_session_from_store("structured stream request start")
                    self.system_prompt = self._build_system_prompt()
                    token_message = self._build_context_token_message()
                    if token_message:
                        info("Memory", f"请求开始 {token_message}")

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在检索长期记忆...",
                        status="running",
                    )
                    self._raise_if_run_cancelled(run_id)
                    relevant_history = self._build_relevant_history_context(user_input)

                    if self.session:
                        if not self._has_user_message_for_run(user_input, run_id):
                            self.session.add_message("user", user_input, run_id=run_id)
                        if self.session_store:
                            with self.runtime.session_lock(self.user_id or "", self.session_id or ""):
                                self.session_store.save_session(self.session)

                    context_messages = self._get_context_messages()
                    section(f"完整上下文 | rid={run_id[:8]}")
                    info("Context", f"[SYSTEM PROMPT] {self.system_prompt[:200]}")
                    info("Context", f"[USER MESSAGES] 共 {len(context_messages)} 条")
                    for i, msg in enumerate(context_messages):
                        info("Context", f"  {i+1}. [{msg['role']}] {str(msg['content'])[:140]}")
                    section("END 上下文")

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在执行 Agent 工作流...",
                        status="running",
                    )
                    self._raise_if_run_cancelled(run_id)
                    checkpoint_state = None
                    checkpoint_start_node = "llm"
                    if resume:
                        checkpoint_state, checkpoint_start_node = self._state_from_checkpoint(
                            self.runtime.get_latest_run_checkpoint(run_id),
                            event_queue,
                        )
                        if checkpoint_state:
                            await self._emit_stream_event(
                                event_queue,
                                "status",
                                run_id,
                                f"已加载任务恢复点，准备从 {checkpoint_start_node} 继续...",
                                status="running",
                            )

                    initial_state = checkpoint_state or {
                        "messages": context_messages,
                        "tool_calls": [],
                        "tool_call_chain": [],
                        "final_response": "",
                        "request_id": run_id,
                        "retrieval_context": relevant_history,
                        "event_queue": event_queue,
                        "checkpoint_enabled": True,
                    }

                    result = await self._run_resumable_workflow(
                        initial_state,
                        checkpoint_start_node if checkpoint_state else "llm",
                    )
                    self._raise_if_run_cancelled(run_id)
                    final_response = result["final_response"]
                    final_response_holder["text"] = final_response
                    update_langfuse_observation(
                        observation,
                        output=final_response,
                        metadata=self._langfuse_completion_metadata(run_id, "stream", result),
                    )

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在输出最终回答...",
                        status="running",
                    )
                    if final_response:
                        for char in final_response:
                            self._raise_if_run_cancelled(run_id)
                            await self._emit_stream_event(
                                event_queue,
                                "content",
                                run_id,
                                content=char,
                            )
                            partial_response_holder["text"] += char
                            if output_delay > 0:
                                await asyncio.sleep(output_delay)

                    self._raise_if_run_cancelled(run_id)
                    if self.session and final_response:
                        self.session.add_message("assistant", final_response, run_id=run_id)

                    if self.session_store and self.session:
                        await self._emit_stream_event(
                            event_queue,
                            "status",
                            run_id,
                            "正在保存会话...",
                            status="running",
                        )
                        self._raise_if_run_cancelled(run_id)
                        with self.runtime.session_lock(self.user_id or "", self.session_id or ""):
                            self.session_store.save_session(self.session)

                    if self.consolidator and self.session:
                        self._raise_if_run_cancelled(run_id)
                        should_consolidate = self.consolidator.should_consolidate(
                            self.session,
                            self.system_prompt,
                        )
                        if should_consolidate:
                            queued = self.runtime.enqueue_task(
                                "memory.consolidate",
                                {
                                    "run_id": run_id,
                                    "user_id": self.user_id,
                                    "session_id": self.session_id,
                                    "message_count": len(self.session.messages),
                                    "last_consolidated": self.session.last_consolidated,
                                    "system_prompt": self.system_prompt,
                                    "context_window_tokens": self.consolidator.context_window_tokens,
                                    "max_completion_tokens": self.consolidator.max_completion_tokens,
                                    "safety_buffer": self.consolidator.safety_buffer,
                                },
                            )
                            if queued:
                                self._raise_if_run_cancelled(run_id)
                                await self._emit_stream_event(
                                    event_queue,
                                    "status",
                                    run_id,
                                    "长期记忆归档已进入后台队列",
                                    status="queued",
                                )
                            else:
                                await self._emit_stream_event(
                                    event_queue,
                                    "status",
                                    run_id,
                                    "正在归档长期记忆...",
                                    status="running",
                                )
                                self._raise_if_run_cancelled(run_id)
                                await self.consolidator.consolidate(
                                    self.session,
                                    self.system_prompt,
                                    request_id=run_id,
                                )

                    self._raise_if_run_cancelled(run_id)
                    token_message = self._build_context_token_message()
                    if token_message:
                        info("Memory", token_message)
                        await self._emit_stream_event(
                            event_queue,
                            "status",
                            run_id,
                            token_message,
                            status="completed",
                        )

                    self.runtime.record_run_complete(run_id, "completed", final_response)
                    await self._emit_stream_event(
                        event_queue,
                        "done",
                        run_id,
                        "本轮任务完成",
                        status="completed",
                        done=True,
                    )
                    ok("Agent", f"完成 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={run_id[:8]}")
                except asyncio.CancelledError as e:
                    reason = str(e) or "用户主动停止任务"
                    update_langfuse_observation(
                        observation,
                        level="WARNING",
                        status_message=reason,
                    )
                    self.runtime.record_run_complete(
                        run_id,
                        "cancelled",
                        partial_response_holder["text"],
                        reason,
                    )
                    self.runtime.record_run_step(
                        run_id=run_id,
                        step_type="cancelled",
                        status="cancelled",
                        message=reason,
                    )
                    ok("Agent", f"已停止 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={run_id[:8]}")
                except Exception as e:
                    update_langfuse_observation(
                        observation,
                        level="ERROR",
                        status_message=str(e),
                    )
                    self.runtime.record_run_complete(
                        run_id,
                        "failed",
                        partial_response_holder["text"] or final_response_holder["text"],
                        str(e),
                    )
                    await self._emit_stream_event(
                        event_queue,
                        "error",
                        run_id,
                        str(e),
                        status="failed",
                    )
                finally:
                    flush_langfuse()
                    await event_queue.put(sentinel)

        task = asyncio.create_task(run_workflow())
        self._mark_run_active(run_id, task, self.user_id or "", self.session_id or "")
        try:
            while True:
                event = await event_queue.get()
                if event is sentinel:
                    break
                yield event
            await task
        finally:
            if task.done():
                self._mark_run_inactive(run_id)
            else:
                task.add_done_callback(
                    lambda _task, active_run_id=run_id: self._mark_run_inactive(active_run_id)
                )

    async def chat(self, user_input: str) -> str:
        """
        处理用户输入（非流式）
        """
        request_id = str(uuid.uuid4())[:8]

        self._refresh_session_from_store("chat request start")

        # 构建系统提示词
        self.system_prompt = self._build_system_prompt()
        token_message = self._build_context_token_message()
        if token_message:
            info("Memory", f"请求开始 {token_message}")

        # 获取相关历史，仅作为本轮临时上下文传给 LLM
        relevant_history = self._build_relevant_history_context(user_input)

        # 添加到会话：只持久化用户原始输入，避免把检索上下文写入 session.jsonl
        if self.session:
            self.session.add_message("user", user_input)

        # 构建上下文并执行（从 session 获取未整合的历史，包含当前用户输入）
        context_messages = self._get_context_messages()

        graph = self.create_graph()
        initial_state = {
            "messages": context_messages,
            "tool_calls": [],
            "tool_call_chain": [],
            "final_response": "",
            "request_id": request_id,
            "retrieval_context": relevant_history,
        }

        with langfuse_request_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            input_data=user_input,
            metadata={"mode": "sync"},
        ) as observation:
            try:
                result = await graph.ainvoke(initial_state)
                final_response = result["final_response"]
                update_langfuse_observation(
                    observation,
                    output=final_response,
                    metadata=self._langfuse_completion_metadata(request_id, "sync", result),
                )

                # 添加助手回复
                if self.session and final_response:
                    self.session.add_message("assistant", final_response)

                # 保存会话
                if self.session_store and self.session:
                    self.session_store.save_session(self.session)

                # 检查整合
                if self.consolidator and self.session:
                    await self.consolidator.maybe_consolidate(
                        self.session,
                        self.system_prompt,
                        request_id=request_id,
                    )

                return final_response
            except Exception as e:
                update_langfuse_observation(
                    observation,
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            finally:
                flush_langfuse()

    async def chat_eval(self, user_input: str, run_id: str | None = None) -> tuple[str, list[str]]:
        """评测专用轻量链路：不写会话、不归档、不记录 run steps，只执行 Agent 主链路。"""
        request_id = run_id or uuid.uuid4().hex
        self.system_prompt = self._build_system_prompt()
        relevant_history = self._build_relevant_history_context(user_input)
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "tool_calls": [],
            "tool_call_chain": [],
            "final_response": "",
            "request_id": request_id,
            "retrieval_context": relevant_history,
            "event_queue": None,
            "checkpoint_enabled": False,
            "record_steps": False,
        }

        with langfuse_request_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            input_data=user_input,
            metadata={"mode": "eval"},
        ) as observation:
            try:
                result = await self._run_resumable_workflow(initial_state, "llm")
                final_response = str(result.get("final_response") or "")
                tool_calls = [
                    str(item.get("tool") or item.get("name"))
                    for item in list(result.get("tool_call_chain") or [])
                    if isinstance(item, dict) and (item.get("tool") or item.get("name"))
                ]
                update_langfuse_observation(
                    observation,
                    output=final_response,
                    metadata=self._langfuse_completion_metadata(request_id, "eval", result),
                )
                return final_response, tool_calls
            except Exception as e:
                update_langfuse_observation(
                    observation,
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            finally:
                flush_langfuse()

    async def start_new_session(self):
        """兼容旧版“新会话”接口：只清空当前短期会话，不强制归档。"""
        if self.session:
            self.session.messages = []
            self.session.last_consolidated = 0
            self.session.updated_at = datetime.now()
            if self.session_store:
                self.session_store.save_session(self.session)

        ok("Agent", "已清空当前会话，未强制归档")
        return {"archived": 0, "message": "已开始新会话（未强制归档）"}

    async def cleanup(self):
        """清理资源"""
        flush_langfuse()
        if self.consolidator:
            try:
                await self.consolidator.close()
            except Exception as exc:
                warn("Agent", f"关闭记忆整合器失败: {exc}")

        close_fn = getattr(self.llm_client, "close", None) or getattr(self.llm_client, "aclose", None)
        if close_fn:
            try:
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                warn("Agent", f"关闭 LLM 客户端失败: {exc}")

        if self._owns_mcp_client:
            await self.mcp_client.disconnect()
"""
定义了 Agent 的核心逻辑：LLM 推理、工具调用、记忆管理、子代理、支持流式输出
"""

import json
import inspect
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, TypedDict, Sequence
from observability import (
    flush_langfuse,
    get_langfuse_metadata,
    init_langfuse,
    langfuse_request_trace,
    langfuse_tool_span,
    update_langfuse_observation,
)

init_langfuse()
from langfuse.openai import AsyncOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from mcp_client import MCPClient
from config import (
    MODEL_CONFIG,
    SUBAGENT_CONFIG,
    CLARIFICATION_CONFIG,
    AGENT_CONFIG,
    CONCURRENCY_CONFIG,
    LOCAL_TOOL_CONFIG,
)
from industrial_runtime import get_industrial_runtime, sanitize_payload, safe_json
from session_store import SessionStore, SessionData
from memory_manager import VectorHistoryStore
from memory_consolidator import MemoryConsolidator
from skills_loader import SkillsLoader
from tools.registry import ToolRegistry
from tools.shell import ExecTool
from logger import error, info, ok, section, warn
from tools.filesystem import ReadFileTool
from tools.clarification import AskClarificationTool
from tools.web import WebFetchTool, WebSearchTool
from subagents import SubAgentRunner, SubAgentConfig, DEFAULT_SUBAGENTS, TASK_TOOL_SYSTEM_PROMPT
from tool_guard import ToolCallGuard
import asyncio
import threading
import uuid

# prompt 文件目录
_PROMPT_DIR = Path(__file__).parent / "prompt"

# 定义状态
class AgentState(TypedDict):
    messages: Sequence[dict]  # 对话历史
    tool_calls: list  # 当前需要调用的工具
    tool_call_chain: list  # 本轮已执行的完整工具调用链
    final_response: str
    request_id: str
    retrieval_context: str
    event_queue: Any


class TravelAgent:
    """Travel Agent - 支持工具调用和长期记忆"""

    _active_run_ids: ClassVar[set[str]] = set()
    _active_run_ids_lock: ClassVar[threading.Lock] = threading.Lock()
    _active_run_tasks: ClassVar[dict[str, asyncio.Task]] = {}
    _active_run_sessions: ClassVar[dict[str, tuple[str, str]]] = {}
    _active_run_tasks_lock: ClassVar[threading.Lock] = threading.Lock()
    _llm_semaphore: ClassVar[asyncio.Semaphore | None] = None
    _llm_semaphore_limit: ClassVar[int] = 0

    def __init__(self, enable_memory: bool = True, session_id: str = None, user_id: str = None, skills_info: str = "",
                 context_window_tokens: int = 9000, max_completion_tokens: int = 1024,
                 workspace: str = None, mcp_client: MCPClient | None = None):
        self.mcp_client = mcp_client or MCPClient()
        self._owns_mcp_client = mcp_client is None
        self.llm_client = AsyncOpenAI(
            api_key=MODEL_CONFIG["api_key"],
            base_url=MODEL_CONFIG["base_url"]
        )
        self.model = MODEL_CONFIG["model_name"]
        self.system_prompt = ""
        self._graph = None
        self.max_workflow_iterations = AGENT_CONFIG["max_iterations"]
        self.tool_guard = ToolCallGuard()
        self._static_prompt_cache_key = None
        self._static_prompt_before_memory = ""
        self._static_prompt_after_memory = ""
        self.session_id = session_id
        self.user_id = user_id or session_id
        self.runtime = get_industrial_runtime()

        # 初始化 SkillsLoader
        self.skills_loader = SkillsLoader()
        self.skills_info = skills_info  # 外部传入的 skill 元数据（优先级更高）

        # 上下文窗口配置
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens

        # 初始化记忆系统
        self.enable_memory = enable_memory
        self.session_store = None
        self.vector_store = None
        self.consolidator = None
        self.session = None  # SessionData 对象

        # 初始化本地工具注册表（用于 skill 执行）
        self.tools = ToolRegistry()
        self._register_default_tools(workspace)

        # 初始化子代理运行器
        self.subagent_runner = None
        if SUBAGENT_CONFIG.get("enabled", True):
            subagent_configs = [
                SubAgentConfig(
                    name=cfg["name"],
                    description=cfg["description"],
                    system_prompt=cfg["system_prompt"],
                    max_iterations=cfg.get("max_iterations", 8),
                )
                for cfg in SUBAGENT_CONFIG.get("subagents", [])
            ] if SUBAGENT_CONFIG.get("subagents") else DEFAULT_SUBAGENTS

            self.subagent_runner = SubAgentRunner(
                llm_client=self.llm_client,
                model=self.model,
                mcp_client=self.mcp_client,
                tools=self.tools,
                subagent_configs=subagent_configs,
            )
            info("Agent", f"子代理已启用 | types={list(self.subagent_runner.subagent_configs.keys())}")

        if enable_memory and self.user_id and session_id:
            try:
                # 初始化会话存储
                self.session_store = SessionStore()
                # 加载或创建会话
                self.session = self.session_store.load_session(self.user_id, session_id)

                # 初始化向量历史存储
                self.vector_store = VectorHistoryStore(self.user_id)
                # 加载并索引现有 HISTORY.md
                history_content = self.session_store.read_history_md(self.user_id)
                if history_content.strip() != "# Conversation History\n\n":
                    self.vector_store.index_history(history_content)

                # 初始化记忆整合器（预算驱动）
                self.consolidator = MemoryConsolidator(
                    session_store=self.session_store,
                    vector_store=self.vector_store,
                    user_id=self.user_id,
                    context_window_tokens=context_window_tokens,
                    max_completion_tokens=max_completion_tokens,
                    safety_buffer=1024
                )

                ok("Agent", f"记忆系统初始化完成 | uid={self.user_id[:8]} sid={session_id[:8]}")
            except Exception as e:
                warn("Agent", f"记忆系统初始化失败: {e}")
                self.enable_memory = False

    def _register_default_tools(self, workspace: str = None) -> None:
        """注册默认的工具集，用于技能执行。"""
        # 设置工作目录
        if workspace is None:
            workspace = os.getcwd()
        workspace_path = Path(workspace)

        # 文件读取仅允许项目目录和 skills 目录，避免公网请求读取主机敏感文件。
        skills_dir = self.skills_loader.skills_dir if hasattr(self.skills_loader, 'skills_dir') else None
        extra_dirs = [skills_dir] if skills_dir else None
        if LOCAL_TOOL_CONFIG.get("read_file_enabled", True):
            self.tools.register(ReadFileTool(
                workspace=workspace_path,
                allowed_dir=workspace_path,
                extra_allowed_dirs=extra_dirs,
            ))

        # 内置只读联网工具。无需外部 MCP 或额外 API Key；所有 URL 读取均经过
        # 公网地址校验、端口限制、超时和响应大小限制。
        if LOCAL_TOOL_CONFIG.get("web_enabled", True):
            self.tools.register(WebSearchTool(
                endpoint=LOCAL_TOOL_CONFIG.get("web_search_endpoint"),
                timeout=LOCAL_TOOL_CONFIG.get("web_timeout_seconds", 15),
            ))
            self.tools.register(WebFetchTool(
                timeout=LOCAL_TOOL_CONFIG.get("web_timeout_seconds", 15),
            ))

        # 公网部署默认不注册 Shell；只有显式开启时才启用沙箱执行。
        if LOCAL_TOOL_CONFIG.get("shell_enabled", True):
            self.tools.register(ExecTool(
                working_dir=str(workspace_path),
                use_sandbox=True,
                sandbox_config=None  # 自动搜索 .srt-settings.json
            ))

        # 注册 ask_clarification 工具（缺信息时主动向用户澄清）
        if CLARIFICATION_CONFIG.get("enabled", True):
            self.tools.register(AskClarificationTool())

        info("Agent", f"已注册本地工具: {self.tools.tool_names}")

    @classmethod
    def _get_llm_semaphore(cls) -> asyncio.Semaphore | None:
        """按进程共享 LLM 并发阀门，避免高并发评测时把模型服务打爆。"""
        limit = int(CONCURRENCY_CONFIG.get("llm_limit") or 0)
        if limit <= 0:
            return None
        if cls._llm_semaphore is None or cls._llm_semaphore_limit != limit:
            cls._llm_semaphore = asyncio.Semaphore(limit)
            cls._llm_semaphore_limit = limit
        return cls._llm_semaphore

    async def _create_llm_completion(self, **kwargs):
        """统一包一层 LLM 调用，方便按环境变量控制全局并发。"""
        semaphore = self._get_llm_semaphore()
        if semaphore is None:
            return await self.llm_client.chat.completions.create(**kwargs)
        async with semaphore:
            return await self.llm_client.chat.completions.create(**kwargs)

    def _build_system_prompt_uncached(self) -> str:
        """构建系统提示词（参考 nanobot 的 ContextBuilder）"""
        # 获取 MCP 工具描述
        mcp_tools_desc = self.mcp_client.get_tools_description()

        # 获取本地工具描述（用于 skill 执行）
        local_tools_desc = self._get_local_tools_description()

        # 获取 MEMORY.md 内容
        memory_content = ""
        if self.session_store and self.user_id:
            memory_content = self.session_store.read_memory_md(self.user_id)

        # 构建提示词（按照 nanobot 的顺序）
        parts = []

        # 1. 主 agent 提示词（身份 + 工作流程）
        parts.append(self._get_base_system_prompt())

        # 2. MCP 可用工具
        parts.append(f"## MCP 可用工具\n\n{mcp_tools_desc}")

        # 3. 本地工具
        if local_tools_desc:
            parts.append(f"## 本地工具\n\n{local_tools_desc}")

        # 4. task 子代理工具
        if self.subagent_runner:
            task_desc = self.subagent_runner.get_task_tool_description()
            parts.append(f"## 子代理工具\n\n- **task**: {task_desc}")

        # 5. Skill 信息（渐进式加载）
        skills_section = self._build_skills_section()
        if skills_section:
            parts.append(skills_section)

        # 6. 长期记忆（仅当内容非空时）
        if memory_content and memory_content.strip():
            parts.append(f"## 长期记忆\n\n{memory_content}")

        # 7. 子代理使用指南
        if self.subagent_runner:
            subagent_guide = f"{TASK_TOOL_SYSTEM_PROMPT}\n可用子代理类型：\n{self.subagent_runner.get_available_subagents_desc()}"
            parts.append(subagent_guide)

        return "\n\n---\n\n".join(parts)

    def _get_static_system_prompt_parts(self) -> tuple[str, str]:
        cache_key = (
            getattr(self.mcp_client, "tools_version", 0),
            self.skills_info,
            tuple(self.tools.tool_names),
            bool(self.subagent_runner),
        )
        if cache_key == self._static_prompt_cache_key:
            return self._static_prompt_before_memory, self._static_prompt_after_memory

        before_memory = []
        after_memory = []

        mcp_tools_desc = self.mcp_client.get_tools_description()
        local_tools_desc = self._get_local_tools_description()

        before_memory.append(self._get_base_system_prompt())
        before_memory.append(f"## MCP 可用工具\n\n{mcp_tools_desc}")

        if local_tools_desc:
            before_memory.append(f"## 本地工具\n\n{local_tools_desc}")

        if self.subagent_runner:
            task_desc = self.subagent_runner.get_task_tool_description()
            before_memory.append(f"## 子代理工具\n\n- **task**: {task_desc}")

        skills_section = self._build_skills_section()
        if skills_section:
            before_memory.append(skills_section)

        if self.subagent_runner:
            subagent_guide = f"{TASK_TOOL_SYSTEM_PROMPT}\n可用子代理类型：\n{self.subagent_runner.get_available_subagents_desc()}"
            after_memory.append(subagent_guide)

        self._static_prompt_cache_key = cache_key
        self._static_prompt_before_memory = "\n\n---\n\n".join(before_memory)
        self._static_prompt_after_memory = "\n\n---\n\n".join(after_memory)
        return self._static_prompt_before_memory, self._static_prompt_after_memory

    def _build_system_prompt(self) -> str:
        before_memory, after_memory = self._get_static_system_prompt_parts()

        parts = [before_memory]
        memory_content = ""
        if self.session_store and self.user_id:
            memory_content = self.session_store.read_memory_md(self.user_id)

        if memory_content and memory_content.strip():
            parts.append(f"## 长期记忆\n\n{memory_content}")

        if after_memory:
            parts.append(after_memory)

        return "\n\n---\n\n".join(parts)

    def _get_local_tools_description(self) -> str:
        """获取本地工具的描述信息（包含完整参数）"""
        if not self.tools.tool_names:
            return ""

        descriptions = []
        for tool_name in self.tools.tool_names:
            tool = self.tools.get(tool_name)
            if tool:
                desc = f"工具名: {tool_name}\n描述: {tool.description}"
                if tool.parameters and "properties" in tool.parameters:
                    params_desc = []
                    for param_name, param_info in tool.parameters["properties"].items():
                        required = param_name in tool.parameters.get("required", [])
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "无描述")
                        params_desc.append(
                            f"  - {param_name} ({param_type}): {param_desc}{' [必需]' if required else ''}"
                        )
                    if params_desc:
                        desc += f"\n参数:\n{chr(10).join(params_desc)}"
                descriptions.append(desc)
        return "\n\n".join(descriptions)

    _base_system_prompt_cache: ClassVar[str | None] = None

    def _get_base_system_prompt(self) -> str:
        """获取主 agent 完整提示词（身份 + 工作流程，来自 prompt/TRAVEL_AGENT_PROMPT.md）"""
        if TravelAgent._base_system_prompt_cache is None:
            TravelAgent._base_system_prompt_cache = (
                (_PROMPT_DIR / "TRAVEL_AGENT_PROMPT.md")
                .read_text(encoding="utf-8")
                .strip()
            )
        return TravelAgent._base_system_prompt_cache

    def _format_clarification(self, payload: dict, fallback_args: dict) -> str:
        """格式化澄清消息为易读文本。"""
        merged = {
            "question": payload.get("question") or fallback_args.get("question") or "请补充更多信息。",
            "clarification_type": payload.get("clarification_type") or fallback_args.get("clarification_type") or "missing_info",
            "context": payload.get("context") or fallback_args.get("context") or "",
            "options": payload.get("options") or fallback_args.get("options") or [],
        }
        type_icons = {
            "missing_info": "❓",
            "ambiguous_requirement": "❓",
            "approach_choice": "👉",
            "risk_confirmation": "⚠️",
            "suggestion": "💡",
        }
        icon = type_icons.get(merged["clarification_type"], "❓")
        parts: list[str] = []
        if merged["context"]:
            parts.append(f"{icon} {merged['context']}")
            parts.append("")
        parts.append(f"{icon} {merged['question']}")
        if merged["options"]:
            parts.append("")
            for index, option in enumerate(merged["options"], 1):
                parts.append(f"  {index}. {option}")
        return "\n".join(parts)

    def _build_skills_section(self) -> str:
        """构建 Skill 信息部分（参考 nanobot 的 ContextBuilder，渐进式加载）"""
        parts = []

        # 如果外部传入了 skills_info，优先使用
        if self.skills_info:
            parts.append("## 技能信息")
            parts.append(self.skills_info)
            return "\n".join(parts)

        # 否则使用 SkillsLoader 加载
        # 1. 获取所有 skills 的元数据（用于打印）
        all_skills = self.skills_loader.list_skills(filter_unavailable=False)
        if all_skills:
            info("Skills", f"发现 {len(all_skills)} 个 skill")
            for skill in all_skills:
                skill_meta = self.skills_loader._get_skill_meta(skill["name"])
                available = self.skills_loader._check_requirements(skill_meta)
                status = "✓" if available else "✗"
                desc = self.skills_loader._get_skill_description(skill["name"])
                info("Skills", f"  {status} {skill['name']}: {desc[:50]}")

        # 2. 获取 always=true 的 skills 并加载完整内容
        always_skills = self.skills_loader.get_always_skills()
        if always_skills:
            info("Skills", f"自动加载 always=true: {always_skills}")
            always_content = self.skills_loader.load_skills_for_context(always_skills)
            if always_content:
                parts.append("# 激活的技能\n\n这些技能已加载到上下文中，你可以直接使用它们：")
                parts.append(always_content)

        # 3. 获取所有 skills 的元数据摘要
        skills_summary = self.skills_loader.build_skills_summary()
        if skills_summary:
            parts.append("# 可用技能\n")
            parts.append("如需使用某个技能，请使用 read_file 工具读取其 SKILL.md 文件。")
            parts.append("Skills with available=\"false\" 需要先安装依赖。")
            parts.append("")
            parts.append(skills_summary)

        return "\n\n".join(parts) if parts else ""

    def _build_relevant_history_context(self, user_input: str, top_k: int = 2) -> str:
        """构建本轮临时检索上下文；只给 LLM 参考，不写入 session.jsonl。"""
        if not self.vector_store:
            return ""

        results = self.vector_store.search_history(user_input, top_k=top_k)
        if not results:
            return ""

        lines = [
            "## 相关历史记录",
        ]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['full_text']} (相关度: {r['score']:.2f})")

        return "\n".join(lines)

    def _refresh_session_from_store(self, reason: str = "") -> None:
        if not self.session_store or not self.user_id or not self.session_id:
            return

        try:
            old_message_count = len(self.session.messages) if self.session else 0
            old_last_consolidated = (
                self.session.last_consolidated if self.session else 0
            )
            refreshed = self.session_store.load_session(self.user_id, self.session_id)
            self.session = refreshed

            new_message_count = len(refreshed.messages)
            new_last_consolidated = refreshed.last_consolidated
            if (
                new_message_count != old_message_count
                or new_last_consolidated != old_last_consolidated
            ):
                detail = f" | {reason}" if reason else ""
                info(
                    "Memory",
                    "已刷新会话快照"
                    f"{detail} messages={old_message_count}->{new_message_count}"
                    f" last_consolidated={old_last_consolidated}->{new_last_consolidated}",
                )
        except Exception as exc:
            warn("Memory", f"刷新会话快照失败: {exc}")

    def _get_context_messages(self) -> list:
        """获取上下文中使用的消息列表（从 session 中获取未整合的历史消息）"""
        messages = []

        # 添加未整合的历史消息（last_consolidated 之后的消息，包含当前用户输入）
        if self.session:
            unconsolidated = self.session.get_unconsolidated_messages()
            for msg in unconsolidated:
                messages.append({
                    "role": msg["role"],
                    "content": msg.get("content", "")
                })

        return messages

    def _tool_display_name(self, tool_name: str) -> str:
        if tool_name == "task":
            return "子 Agent"
        if tool_name == "ask_clarification":
            return "澄清确认"
        return tool_name

    def _summarize_tool_result(self, result: Any, limit: int = 160) -> str:
        text = result if isinstance(result, str) else safe_json(result, limit=limit * 2)
        text = " ".join(str(text).split())
        if len(text) > limit:
            return text[:limit] + "...(truncated)"
        return text

    def _get_mcp_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """读取 MCP 工具注册时提供的 inputSchema，用于调用前硬校验。"""
        tool = getattr(self.mcp_client, "tools", {}).get(tool_name)
        if not tool:
            return None
        schema = getattr(tool, "inputSchema", None)
        return schema if isinstance(schema, dict) else None

    def _prepare_tool_arguments(self, tool_name: str, tool_args: Any) -> tuple[dict[str, Any], str | None]:
        """执行前校验并尽量复用本地工具已有的参数转换逻辑。"""
        if not isinstance(tool_args, dict):
            return {}, f"工具参数校验失败：parameters must be an object, got {type(tool_args).__name__}"

        if tool_name == "task":
            if not str(tool_args.get("task_description") or "").strip():
                return tool_args, "工具参数校验失败：task_description 参数不能为空。"
            errors = self.tool_guard.validate_arguments(tool_name, tool_args, ToolCallGuard.TASK_SCHEMA)
            return tool_args, self._format_tool_validation_error(tool_name, errors) if errors else None

        if self.tools.has(tool_name):
            _tool, cast_params, error_text = self.tools.prepare_call(tool_name, tool_args)
            if error_text:
                return cast_params, f"工具参数校验失败：{error_text}"
            return cast_params, None

        errors = self.tool_guard.validate_arguments(
            tool_name,
            tool_args,
            self._get_mcp_tool_schema(tool_name),
        )
        return tool_args, self._format_tool_validation_error(tool_name, errors) if errors else None

    @staticmethod
    def _format_tool_validation_error(tool_name: str, errors: list[str]) -> str | None:
        """把 schema 校验错误整理成 LLM 可修正的提示。"""
        if not errors:
            return None
        return (
            f"工具参数校验失败：工具 '{tool_name}' 的参数不符合 schema："
            + "; ".join(errors)
            + "。请根据用户上下文补齐或修正参数后重新调用工具。"
        )

    async def _call_mcp_tool_with_retry(self, tool_name: str, tool_args: dict[str, Any], tool_call_record: dict[str, Any]) -> Any:
        """对 MCP 调用增加外层分类重试，参数类错误不会盲目重试。"""
        attempt = 1
        max_attempts = 1
        while True:
            try:
                tool_call_record["attempt"] = attempt
                tool_call_record["max_attempts"] = max_attempts
                return await self.mcp_client.call_tool(tool_name, tool_args)
            except Exception as exc:
                retry_decision = self.tool_guard.classify_exception(exc)
                max_attempts = retry_decision.max_attempts
                tool_call_record.update({
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error_type": retry_decision.error_type,
                    "retryable": retry_decision.retryable,
                })
                if not retry_decision.retryable or attempt >= max_attempts:
                    raise
                delay = 0.5 * (3 ** (attempt - 1))
                warn(
                    "Tool",
                    f"MCP 工具 {tool_name} 调用失败，{delay:.1f}s 后重试 "
                    f"attempt={attempt}/{max_attempts} type={retry_decision.error_type} err={exc}",
                )
                await asyncio.sleep(delay)
                attempt += 1

    def _langfuse_completion_metadata(self, request_id: str, mode: str, result: dict[str, Any]) -> dict[str, Any]:
        tool_call_chain = list(result.get("tool_call_chain") or [])
        tool_calls = [
            str(item.get("tool") or item.get("name"))
            for item in tool_call_chain
            if isinstance(item, dict) and (item.get("tool") or item.get("name"))
        ]
        return get_langfuse_metadata(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            component="travel_agent",
            operation="chat",
            extra={
                "mode": mode,
                "tool_calls": tool_calls,
                "tool_call_chain": tool_call_chain,
                "tool_call_count": len(tool_calls),
            },
        )

    async def _emit_stream_event(
        self,
        event_queue: asyncio.Queue | None,
        event_type: str,
        run_id: str,
        message: str = "",
        **payload: Any,
    ) -> None:
        record_step = bool(payload.pop("record_step", True))
        event = {
            "type": event_type,
            "run_id": run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            **payload,
        }
        if message:
            event["message"] = message

        if event_type != "content" and record_step:
            status = str(payload.get("status") or "running")
            self.runtime.record_run_step(
                run_id=run_id,
                step_type=event_type,
                status=status,
                message=message,
                tool_name=payload.get("tool"),
                payload=sanitize_payload(payload),
                elapsed_ms=payload.get("elapsed_ms"),
            )

        if event_queue is not None:
            await event_queue.put(event)

    async def _emit_state_event(
        self,
        state: AgentState,
        event_type: str,
        message: str = "",
        **payload: Any,
    ) -> None:
        # 轻量评测路径不需要写 run steps，避免高并发评测把 MySQL 写入放大。
        payload.setdefault("record_step", bool(state.get("record_steps", True)))
        await self._emit_stream_event(
            state.get("event_queue"),
            event_type,
            state.get("request_id", ""),
            message,
            **payload,
        )

    def get_context_token_usage(self) -> tuple[int, int] | None:
        if not self.session:
            return None

        if self.consolidator:
            estimated = self.consolidator.estimate_session_tokens(
                self.session,
                self.system_prompt,
            )
            budget = self.consolidator.budget
        else:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self._get_context_messages())
            estimated = sum(len(str(msg.get("content", ""))) for msg in messages) // 3
            budget = self.context_window_tokens - self.max_completion_tokens - 1024
        return estimated, budget

    def _build_context_token_message(self) -> str | None:
        try:
            usage = self.get_context_token_usage()
            if not usage:
                return None
            estimated, budget = usage
            return f"上下文 token: {estimated}/{budget}"
        except Exception as exc:
            warn("Memory", f"上下文 token 统计失败: {exc}")
            return None

    @classmethod
    def is_run_active(cls, run_id: str) -> bool:
        with cls._active_run_ids_lock:
            return run_id in cls._active_run_ids

    @classmethod
    def _mark_run_active(
        cls,
        run_id: str,
        task: asyncio.Task | None = None,
        user_id: str = "",
        session_id: str = "",
    ) -> None:
        with cls._active_run_ids_lock:
            cls._active_run_ids.add(run_id)
        if task is not None:
            with cls._active_run_tasks_lock:
                cls._active_run_tasks[run_id] = task
                if user_id and session_id:
                    cls._active_run_sessions[run_id] = (user_id, session_id)

    @classmethod
    def _mark_run_inactive(cls, run_id: str) -> None:
        with cls._active_run_ids_lock:
            cls._active_run_ids.discard(run_id)
        with cls._active_run_tasks_lock:
            cls._active_run_tasks.pop(run_id, None)
            cls._active_run_sessions.pop(run_id, None)

    @classmethod
    def cancel_run(cls, run_id: str) -> bool:
        with cls._active_run_tasks_lock:
            task = cls._active_run_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    @classmethod
    def cancel_session_run(cls, user_id: str, session_id: str) -> tuple[str, bool]:
        with cls._active_run_tasks_lock:
            active_items = list(cls._active_run_sessions.items())
        for run_id, (active_user_id, active_session_id) in reversed(active_items):
            if active_user_id == user_id and active_session_id == session_id:
                return run_id, cls.cancel_run(run_id)
        return "", False

    def _raise_if_run_cancelled(self, run_id: str) -> None:
        if self.runtime.is_run_cancelled(run_id):
            raise asyncio.CancelledError("用户主动停止任务")

    def _has_user_message_for_run(self, user_input: str, run_id: str) -> bool:
        if not self.session:
            return False

        for msg in self.session.messages:
            if msg.get("role") == "user" and msg.get("run_id") == run_id:
                return True

        # 兼容崩溃前已经落盘、但旧消息缺 run_id 的情况：只把最后一个未回答用户消息视为本轮输入。
        for msg in reversed(self.session.messages):
            role = msg.get("role")
            if role == "assistant":
                return False
            if role == "user":
                return str(msg.get("content") or "") == user_input
        return False

    def _checkpoint_copy(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return sanitize_payload(value, limit=5000)

    def _checkpoint_payload(self, state: AgentState, next_node: str) -> dict[str, Any]:
        return {
            "version": 1,
            "run_id": state.get("request_id", ""),
            "next_node": next_node,
            "messages": self._checkpoint_copy(list(state.get("messages") or [])),
            "tool_calls": self._checkpoint_copy(list(state.get("tool_calls") or [])),
            "tool_call_chain": self._checkpoint_copy(list(state.get("tool_call_chain") or [])),
            "final_response": str(state.get("final_response") or ""),
            "retrieval_context": str(state.get("retrieval_context") or ""),
        }

    def _save_run_checkpoint(self, state: AgentState, next_node: str) -> None:
        if not state.get("checkpoint_enabled"):
            return
        run_id = str(state.get("request_id") or "")
        if not run_id:
            return
        try:
            self.runtime.record_run_checkpoint(run_id, self._checkpoint_payload(state, next_node))
        except Exception as exc:
            warn("Agent", f"保存任务恢复点失败 | rid={run_id[:8]} err={exc}")

    def _state_from_checkpoint(
        self,
        checkpoint: dict[str, Any] | None,
        event_queue: asyncio.Queue | None,
    ) -> tuple[AgentState | None, str]:
        if not isinstance(checkpoint, dict):
            return None, "llm"
        next_node = str(checkpoint.get("next_node") or "llm")
        if next_node not in {"llm", "tool", "end"}:
            return None, "llm"

        messages = checkpoint.get("messages")
        tool_calls = checkpoint.get("tool_calls")
        if not isinstance(messages, list) or not isinstance(tool_calls, list):
            return None, "llm"

        state: AgentState = {
            "messages": messages,
            "tool_calls": tool_calls,
            "tool_call_chain": list(checkpoint.get("tool_call_chain") or []),
            "final_response": str(checkpoint.get("final_response") or ""),
            "request_id": str(checkpoint.get("run_id") or ""),
            "retrieval_context": str(checkpoint.get("retrieval_context") or ""),
            "event_queue": event_queue,
            "checkpoint_enabled": True,
        }
        if not state["request_id"]:
            return None, "llm"
        return state, next_node

    def _merge_command_update(self, state: AgentState, command: Command) -> AgentState:
        update = getattr(command, "update", None)
        if not isinstance(update, dict):
            return state

        for key, value in update.items():
            if key == "messages":
                messages = list(state.get("messages") or [])
                if isinstance(value, list):
                    messages.extend(value)
                elif value:
                    messages.append(value)
                state["messages"] = messages
            elif key in {"tool_calls", "tool_call_chain", "final_response", "retrieval_context"}:
                state[key] = value
            else:
                state[key] = value
        return state

    def _next_node_from_command(self, state: AgentState, command: Command) -> str:
        goto = getattr(command, "goto", None)
        if goto == END or goto == "__end__":
            return "end"
        if isinstance(goto, str) and goto in {"llm", "tool"}:
            return str(goto)
        if state.get("final_response"):
            return "end"
        return "llm"

    async def _run_resumable_workflow(
        self,
        initial_state: AgentState,
        start_node: str = "llm",
        max_iterations: int | None = None,
    ) -> AgentState:
        state = initial_state
        node = start_node if start_node in {"llm", "tool", "end"} else "llm"
        iteration_limit = max_iterations or self.max_workflow_iterations

        for _ in range(iteration_limit):
            if node == "end":
                self._save_run_checkpoint(state, "end")
                return state

            if node == "llm":
                state = await self.llm_node(state)
                node = self.should_continue(state)
                if node == "end":
                    self._save_run_checkpoint(state, "end")
                    return state
                continue

            if node == "tool":
                tool_result = await self.tool_node(state)
                if hasattr(tool_result, "update") and hasattr(tool_result, "goto"):
                    state = self._merge_command_update(state, tool_result)
                    node = self._next_node_from_command(state, tool_result)
                    self._save_run_checkpoint(state, node)
                    if node == "end":
                        return state
                    continue

                state = tool_result
                node = self._should_continue_after_tool(state)
                if node == "end":
                    self._save_run_checkpoint(state, "end")
                    return state
                continue

        raise RuntimeError(f"Agent 工作流超过最大迭代次数: {iteration_limit}")

    async def llm_node(self, state: AgentState) -> AgentState:
        """
        LLM推理节点
        """
        # 构建完整消息列表
        await self._emit_state_event(state, "status", "正在思考下一步...", status="running")
        system_prompt = self.system_prompt
        retrieval_context = state.get("retrieval_context", "").strip()
        if retrieval_context:
            system_prompt = f"{system_prompt}\n\n---\n\n{retrieval_context}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(state["messages"])

        # 调用 LLM
        response = await self._create_llm_completion(
            model=self.model,
            messages=messages,
            temperature=0.1,
            name="travel-agent-llm",
            metadata=get_langfuse_metadata(
                session_id=self.session_id,
                user_id=self.user_id,
                request_id=state.get("request_id"),
                component="travel_agent",
                operation="llm_node",
            ),
        )
        content = response.choices[0].message.content.strip()
        state["messages"].append({"role": "assistant", "content": content})

        # 解析工具调用
        tool_call = self._parse_tool_call(content)

        info("LLM", f"响应: {content[:120]}")
        info("LLM", f"解析工具调用: {tool_call}")

        if tool_call:
            await self._emit_state_event(
                state,
                "status",
                f"准备调用 {self._tool_display_name(tool_call.get('tool', 'unknown'))} 工具...",
                status="running",
                tool=tool_call.get("tool"),
            )
            state["tool_calls"] = [tool_call]
            state["final_response"] = ""
        else:
            await self._emit_state_event(state, "status", "正在整理最终回答...", status="running")
            state["final_response"] = content
            state["tool_calls"] = []

        self._save_run_checkpoint(
            state,
            "tool" if state.get("tool_calls") else "end",
        )
        return state

    async def tool_node(self, state: AgentState) -> AgentState | Command:
        """
        工具调用节点
        1. 如果是 ask_clarification → 返回 Command(goto=END) 中断本轮
        2. 如果是 task 工具 → 委派给子代理
        3. 否则如果是本地工具（exec, read_file）→ 直接执行
        4. 否则调用 MCP 工具
        """
        if not state.get("tool_calls"):
            return state

        tool_call = state["tool_calls"][0]
        tool_name = tool_call["tool"]
        tool_args = tool_call["arguments"]
        tool_args, validation_error = self._prepare_tool_arguments(tool_name, tool_args)
        display_name = self._tool_display_name(tool_name)
        sanitized_args = sanitize_payload(tool_args)
        tool_call_chain = state.setdefault("tool_call_chain", [])
        signature = self.tool_guard.signature(tool_name, tool_args)

        if validation_error:
            tool_call_record = {
                "tool": tool_name,
                "arguments": sanitized_args,
                "signature": signature,
                "status": "failed",
                "error_type": "validation",
                "retryable": False,
                "error": validation_error,
            }
            tool_call_chain.append(tool_call_record)
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具参数校验失败",
                status="failed",
                tool=tool_name,
                result_summary=validation_error,
            )
            state["messages"].append({"role": "user", "content": validation_error})
            if self.tool_guard.should_stop_after_failure(tool_call_chain):
                state["final_response"] = self.tool_guard.build_failure_stop_response(tool_call_chain)
            state["tool_calls"] = []
            self._save_run_checkpoint(
                state,
                "end" if state.get("final_response") else "llm",
            )
            return state

        guard_decision = self.tool_guard.before_call(tool_call_chain, tool_name, tool_args)
        if guard_decision.action in {"skip", "terminate"}:
            tool_call_record = {
                "tool": tool_name,
                "arguments": sanitized_args,
                "signature": guard_decision.signature,
                "attempt": guard_decision.attempt,
                "status": "blocked" if guard_decision.action == "terminate" else "skipped",
                "result_summary": guard_decision.last_result_summary or guard_decision.message,
            }
            tool_call_chain.append(tool_call_record)
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具重复调用已拦截",
                status=tool_call_record["status"],
                tool=tool_name,
                result_summary=guard_decision.final_response or guard_decision.message,
            )
            if guard_decision.action == "terminate":
                state["final_response"] = guard_decision.final_response
            else:
                state["messages"].append({"role": "user", "content": guard_decision.message})
            state["tool_calls"] = []
            self._save_run_checkpoint(
                state,
                "end" if state.get("final_response") else "llm",
            )
            return state

        tool_call_record = {
            "tool": tool_name,
            "arguments": sanitized_args,
            "signature": signature,
            "attempt": guard_decision.attempt,
            "status": "running",
        }
        tool_call_chain.append(tool_call_record)
        if tool_name == "ask_clarification":
            started_at = time.perf_counter()
            await self._emit_state_event(
                state,
                "tool_start",
                f"正在调用 {display_name} 工具...",
                status="running",
                tool=tool_name,
                arguments=sanitized_args,
            )

        # 拦截 ask_clarification：返回 Command(goto=END) 中断本轮循环
        if tool_name == "ask_clarification":
            info("Tool", "调用澄清工具 ask_clarification")
            try:
                result_str = await self.tools.execute(tool_name, tool_args)
                payload = json.loads(result_str) if isinstance(result_str, str) else {}
            except Exception as exc:
                payload = {}
                tool_call_record.update({"status": "failed", "error": str(exc)})
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            if tool_call_record.get("status") != "failed":
                tool_call_record.update({
                    "status": "success",
                    "elapsed_ms": elapsed_ms,
                    "result_summary": self._summarize_tool_result(payload),
                })
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具已完成",
                status="success",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=self._summarize_tool_result(payload),
            )
            formatted = self._format_clarification(payload, tool_args)
            checkpoint_state = dict(state)
            checkpoint_state["messages"] = list(state.get("messages") or []) + [
                {"role": "assistant", "content": formatted}
            ]
            checkpoint_state["final_response"] = formatted
            checkpoint_state["tool_calls"] = []
            self._save_run_checkpoint(checkpoint_state, "end")
            return Command(
                update={
                    "messages": [{"role": "assistant", "content": formatted}],
                    "final_response": formatted,
                    "tool_calls": [],
                },
                goto=END,
            )

        started_at = time.perf_counter()
        await self._emit_state_event(
            state,
            "tool_start",
            f"正在调用 {display_name} 工具...",
            status="running",
            tool=tool_name,
            arguments=sanitized_args,
        )

        tool_span_cm = langfuse_tool_span(
            tool_name=tool_name,
            arguments=sanitized_args,
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=state.get("request_id"),
        )
        tool_span = tool_span_cm.__enter__()

        try:
            # 1. 检查是否为 task 子代理工具
            if tool_name == "task" and self.subagent_runner:
                info("Tool", f"调用子代理工具 task | 参数={tool_args}")
                subagent_type = tool_args.get("subagent_type", "general-purpose")
                task_description = tool_args.get("task_description", "")
                if not task_description:
                    result = "错误：task_description 参数不能为空"
                else:
                    result = await self.subagent_runner.run(
                        subagent_type,
                        task_description,
                        session_id=self.session_id,
                        user_id=self.user_id,
                        request_id=state.get("request_id"),
                    )
                info("Tool", f"子代理结果: {result[:120]}")

            # 2. 优先检查本地工具（用于 skill 执行）
            elif self.tools.has(tool_name):
                info("Tool", f"调用本地工具 {tool_name} | 参数={tool_args}")
                result = await self.tools.execute(tool_name, tool_args)
                info("Tool", f"结果: {str(result)[:120]}")
            # 3. 否则调用 MCP 工具
            else:
                info("Tool", f"调用 MCP 工具 {tool_name}")
                result = await self._call_mcp_tool_with_retry(tool_name, tool_args, tool_call_record)

            # 工具以 Error/错误 开头时视为失败，避免把失败结果当作有效新增信息继续循环。
            if isinstance(result, str) and result.lstrip().lower().startswith("error"):
                raise RuntimeError(result)
            if isinstance(result, str) and result.lstrip().startswith("错误"):
                raise RuntimeError(result)

            update_langfuse_observation(
                tool_span,
                output=self._summarize_tool_result(result, limit=500),
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            tool_call_record.update({
                "status": "success",
                "elapsed_ms": elapsed_ms,
                "result_summary": self._summarize_tool_result(result),
            })

            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具已返回",
                status="success",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=self._summarize_tool_result(result),
            )

            result_message = (
                f"[工具执行完成]\n"
                f"工具名称: {tool_name}\n"
                f"输入参数: {tool_args}\n"
                f"执行结果: {result}\n\n"
                f"请根据以上结果：\n"
                f"- 如果需要更多信息，调用其他工具\n"
                f"- 如果信息已充分，用中文向用户总结答案"
            )
            state["messages"].append({"role": "user", "content": result_message})
        except RuntimeError as e:
            update_langfuse_observation(tool_span, level="ERROR", status_message=str(e))
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            tool_call_record.update({
                "status": "failed",
                "elapsed_ms": elapsed_ms,
                "error": str(e),
            })
            if "重连失败" in str(e) or "连接异常" in str(e):
                error_message = (
                    f"工具'{tool_name}'暂时不可用（连接已断开且重连失败）。\n"
                    f"请尝试：\n"
                    f"1. 使用其他可用工具完成任务\n"
                    f"2. 或向用户说明情况并建议稍后重试"
                )
            else:
                error_message = f"工具'{tool_name}'执行失败: {str(e)}\n请尝试其他方法或向用户说明。"
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具调用失败",
                status="failed",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=str(e),
            )
            state["messages"].append({"role": "user", "content": error_message})
            if self.tool_guard.should_stop_after_failure(tool_call_chain):
                state["final_response"] = self.tool_guard.build_failure_stop_response(tool_call_chain)
        except Exception as e:
            update_langfuse_observation(tool_span, level="ERROR", status_message=str(e))
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            tool_call_record.update({
                "status": "failed",
                "elapsed_ms": elapsed_ms,
                "error": str(e),
            })
            error_message = f"工具'{tool_name}'执行失败: {str(e)}\n请尝试其他方法或向用户说明。"
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具调用失败",
                status="failed",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=str(e),
            )
            state["messages"].append({"role": "user", "content": error_message})
            if self.tool_guard.should_stop_after_failure(tool_call_chain):
                state["final_response"] = self.tool_guard.build_failure_stop_response(tool_call_chain)
        finally:
            try:
                tool_span_cm.__exit__(None, None, None)
            except Exception:
                pass

        state["tool_calls"] = []
        self._save_run_checkpoint(state, "end" if state.get("final_response") else "llm")
        return state

    def should_continue(self, state: AgentState) -> str:
        """
        条件判断函数
        """
        if state.get("tool_calls"):
            return "tool"
        elif state.get("final_response"):
            return "end"
        else:
            return "llm"

    def _normalize_tool_call(self, data: Any) -> dict | None:
        """把解析出的 JSON 数据规范化为单个工具调用。"""
        if isinstance(data, dict) and "tool" in data and "arguments" in data:
            if isinstance(data.get("arguments"), dict):
                return data
            return {
                **data,
                "arguments": {},
            }
        if isinstance(data, list):
            for item in data:
                tool_call = self._normalize_tool_call(item)
                if tool_call:
                    return tool_call
        return None

    def _parse_tool_call(self, content: str) -> dict | None:
        """
        解析工具调用
        """
        content = content.strip()
        decoder = json.JSONDecoder()

        # 情况1: 直接是 JSON
        try:
            data = json.loads(content)
            tool_call = self._normalize_tool_call(data)
            if tool_call:
                return tool_call
        except Exception:
            pass

        # 情况2: Markdown 代码块
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                tool_call = self._normalize_tool_call(data)
                if tool_call:
                    return tool_call
            except Exception:
                pass

        # 情况3: 从任意位置扫描 JSON 对象，兼容嵌套 arguments 和一次输出多个 JSON 的情况。
        for match in re.finditer(r"\{", content):
            try:
                data, _ = decoder.raw_decode(content[match.start():])
            except json.JSONDecodeError:
                continue
            tool_call = self._normalize_tool_call(data)
            if tool_call:
                return tool_call

        # 情况4: 查找最外层的大括号
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and start < end:
            try:
                data = json.loads(content[start:end + 1])
                tool_call = self._normalize_tool_call(data)
                if tool_call:
                    return tool_call
            except Exception:
                pass

        return None

    def create_graph(self):
        """
        创建 LangGraph 工作流
        """
        if self._graph is not None:
            return self._graph

        workflow = StateGraph(AgentState)
        workflow.add_node("llm", self.llm_node)
        workflow.add_node("tool", self.tool_node)
        workflow.add_edge(START, "llm")
        workflow.add_conditional_edges(
            "llm",
            self.should_continue,
            {"tool": "tool", "end": END, "llm": "llm"}
        )
        workflow.add_conditional_edges(
            "tool",
            self._should_continue_after_tool,
            {"end": END, "llm": "llm"}
        )

        self._graph = workflow.compile()
        return self._graph

    def _should_continue_after_tool(self, state: AgentState) -> str:
        """
        工具执行后的条件判断
        - 如果 final_response 已设置（如 ask_clarification），则结束
        - 否则继续 LLM 推理
        """
        if state.get("final_response"):
            return "end"
        else:
            return "llm"

    async def chat_stream(self, user_input: str):
        """
        流式处理用户输入（新版）
        """
        request_id = str(uuid.uuid4())[:8]
        info("Agent", f"开始处理 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={request_id}")

        self._refresh_session_from_store("stream request start")

        # 1. 构建系统提示词
        self.system_prompt = self._build_system_prompt()
        token_message = self._build_context_token_message()
        if token_message:
            info("Memory", f"请求开始 {token_message}")

        # 2. 获取相关历史（向量搜索），仅作为本轮临时上下文传给 LLM
        relevant_history = self._build_relevant_history_context(user_input)

        # 3. 添加到会话：只持久化用户原始输入，避免把检索上下文写入 session.jsonl
        if self.session:
            self.session.add_message("user", user_input)

        # 4. 构建上下文消息（从 session 获取未整合的历史，包含当前用户输入）
        context_messages = self._get_context_messages()

        # 打印完整上下文
        section(f"完整上下文 | rid={request_id}")
        info("Context", f"[SYSTEM PROMPT] {self.system_prompt[:200]}")
        info("Context", f"[USER MESSAGES] 共 {len(context_messages)} 条:")
        for i, msg in enumerate(context_messages):
            info("Context", f"  {i+1}. [{msg['role']}] {str(msg['content'])[:140]}")
        section("END 上下文")

        # 5. 执行工作流
        graph = self.create_graph()
        initial_state = {
            "messages": context_messages,
            "tool_calls": [],
            "tool_call_chain": [],
            "final_response": "",
            "request_id": request_id,
            "retrieval_context": relevant_history,
        }

        with langfuse_request_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            input_data=user_input,
            metadata={"mode": "stream"},
        ) as observation:
            try:
                info("Agent", f"执行工作流 | rid={request_id}")
                result = await graph.ainvoke(initial_state)
                ok("Agent", f"工作流完成 | rid={request_id}")

                final_response = result["final_response"]
                update_langfuse_observation(
                    observation,
                    output=final_response,
                    metadata=self._langfuse_completion_metadata(request_id, "stream", result),
                )

                # 7. 添加助手回复到会话
                if self.session and final_response:
                    self.session.add_message("assistant", final_response)

                # 8. 保存会话到 JSONL
                if self.session_store and self.session:
                    self.session_store.save_session(self.session)
                    ok("Agent", f"已保存会话 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={request_id}")

                # 9. 流式输出
                if final_response:
                    for char in final_response:
                        yield char
                        await asyncio.sleep(0.02)

                # 10. 检查并执行记忆整合
                if self.consolidator and self.session:
                    await self.consolidator.maybe_consolidate(
                        self.session,
                        self.system_prompt,
                        request_id=request_id,
                    )
            except Exception as e:
                update_langfuse_observation(
                    observation,
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            finally:
                flush_langfuse()

        ok("Agent", f"完成 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={request_id}")

    async def chat_stream(
        self,
        user_input: str,
        run_id: str | None = None,
        resume: bool = False,
        content_delay: float | None = None,
    ):
        """流式输出当前用户请求的结构化运行事件。"""
        run_id = run_id or uuid.uuid4().hex
        # 非流式 /chat 可传 0 跳过逐字等待，前端流式默认保留打字机节奏。
        output_delay = (
            float(CONCURRENCY_CONFIG["stream_content_delay"])
            if content_delay is None
            else max(0.0, float(content_delay))
        )
        event_queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()
        final_response_holder = {"text": ""}
        partial_response_holder = {"text": ""}

        async def run_workflow() -> None:
            info("Agent", f"开始处理 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={run_id[:8]}")
            self.runtime.record_run_start(
                run_id=run_id,
                user_id=self.user_id or "",
                session_id=self.session_id or "",
                input_text=user_input,
            )

            with langfuse_request_trace(
                session_id=self.session_id,
                user_id=self.user_id,
                request_id=run_id,
                input_data=user_input,
                metadata={"mode": "stream"},
            ) as observation:
                try:
                    self._raise_if_run_cancelled(run_id)
                    if resume:
                        await self._emit_stream_event(
                            event_queue,
                            "status",
                            run_id,
                            "服务重启后正在接管任务...",
                            status="running",
                        )

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在构建上下文...",
                        status="running",
                    )
                    self._raise_if_run_cancelled(run_id)
                    self._refresh_session_from_store("structured stream request start")
                    self.system_prompt = self._build_system_prompt()
                    token_message = self._build_context_token_message()
                    if token_message:
                        info("Memory", f"请求开始 {token_message}")

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在检索长期记忆...",
                        status="running",
                    )
                    self._raise_if_run_cancelled(run_id)
                    relevant_history = self._build_relevant_history_context(user_input)

                    if self.session:
                        if not self._has_user_message_for_run(user_input, run_id):
                            self.session.add_message("user", user_input, run_id=run_id)
                        if self.session_store:
                            with self.runtime.session_lock(self.user_id or "", self.session_id or ""):
                                self.session_store.save_session(self.session)

                    context_messages = self._get_context_messages()
                    section(f"完整上下文 | rid={run_id[:8]}")
                    info("Context", f"[SYSTEM PROMPT] {self.system_prompt[:200]}")
                    info("Context", f"[USER MESSAGES] 共 {len(context_messages)} 条")
                    for i, msg in enumerate(context_messages):
                        info("Context", f"  {i+1}. [{msg['role']}] {str(msg['content'])[:140]}")
                    section("END 上下文")

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在执行 Agent 工作流...",
                        status="running",
                    )
                    self._raise_if_run_cancelled(run_id)
                    checkpoint_state = None
                    checkpoint_start_node = "llm"
                    if resume:
                        checkpoint_state, checkpoint_start_node = self._state_from_checkpoint(
                            self.runtime.get_latest_run_checkpoint(run_id),
                            event_queue,
                        )
                        if checkpoint_state:
                            await self._emit_stream_event(
                                event_queue,
                                "status",
                                run_id,
                                f"已加载任务恢复点，准备从 {checkpoint_start_node} 继续...",
                                status="running",
                            )

                    initial_state = checkpoint_state or {
                        "messages": context_messages,
                        "tool_calls": [],
                        "tool_call_chain": [],
                        "final_response": "",
                        "request_id": run_id,
                        "retrieval_context": relevant_history,
                        "event_queue": event_queue,
                        "checkpoint_enabled": True,
                    }

                    result = await self._run_resumable_workflow(
                        initial_state,
                        checkpoint_start_node if checkpoint_state else "llm",
                    )
                    self._raise_if_run_cancelled(run_id)
                    final_response = result["final_response"]
                    final_response_holder["text"] = final_response
                    update_langfuse_observation(
                        observation,
                        output=final_response,
                        metadata=self._langfuse_completion_metadata(run_id, "stream", result),
                    )

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在输出最终回答...",
                        status="running",
                    )
                    if final_response:
                        for char in final_response:
                            self._raise_if_run_cancelled(run_id)
                            await self._emit_stream_event(
                                event_queue,
                                "content",
                                run_id,
                                content=char,
                            )
                            partial_response_holder["text"] += char
                            if output_delay > 0:
                                await asyncio.sleep(output_delay)

                    self._raise_if_run_cancelled(run_id)
                    if self.session and final_response:
                        self.session.add_message("assistant", final_response, run_id=run_id)

                    if self.session_store and self.session:
                        await self._emit_stream_event(
                            event_queue,
                            "status",
                            run_id,
                            "正在保存会话...",
                            status="running",
                        )
                        self._raise_if_run_cancelled(run_id)
                        with self.runtime.session_lock(self.user_id or "", self.session_id or ""):
                            self.session_store.save_session(self.session)

                    if self.consolidator and self.session:
                        self._raise_if_run_cancelled(run_id)
                        should_consolidate = self.consolidator.should_consolidate(
                            self.session,
                            self.system_prompt,
                        )
                        if should_consolidate:
                            queued = self.runtime.enqueue_task(
                                "memory.consolidate",
                                {
                                    "run_id": run_id,
                                    "user_id": self.user_id,
                                    "session_id": self.session_id,
                                    "message_count": len(self.session.messages),
                                    "last_consolidated": self.session.last_consolidated,
                                    "system_prompt": self.system_prompt,
                                    "context_window_tokens": self.consolidator.context_window_tokens,
                                    "max_completion_tokens": self.consolidator.max_completion_tokens,
                                    "safety_buffer": self.consolidator.safety_buffer,
                                },
                            )
                            if queued:
                                self._raise_if_run_cancelled(run_id)
                                await self._emit_stream_event(
                                    event_queue,
                                    "status",
                                    run_id,
                                    "长期记忆归档已进入后台队列",
                                    status="queued",
                                )
                            else:
                                await self._emit_stream_event(
                                    event_queue,
                                    "status",
                                    run_id,
                                    "正在归档长期记忆...",
                                    status="running",
                                )
                                self._raise_if_run_cancelled(run_id)
                                await self.consolidator.consolidate(
                                    self.session,
                                    self.system_prompt,
                                    request_id=run_id,
                                )

                    self._raise_if_run_cancelled(run_id)
                    token_message = self._build_context_token_message()
                    if token_message:
                        info("Memory", token_message)
                        await self._emit_stream_event(
                            event_queue,
                            "status",
                            run_id,
                            token_message,
                            status="completed",
                        )

                    self.runtime.record_run_complete(run_id, "completed", final_response)
                    await self._emit_stream_event(
                        event_queue,
                        "done",
                        run_id,
                        "本轮任务完成",
                        status="completed",
                        done=True,
                    )
                    ok("Agent", f"完成 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={run_id[:8]}")
                except asyncio.CancelledError as e:
                    reason = str(e) or "用户主动停止任务"
                    update_langfuse_observation(
                        observation,
                        level="WARNING",
                        status_message=reason,
                    )
                    self.runtime.record_run_complete(
                        run_id,
                        "cancelled",
                        partial_response_holder["text"],
                        reason,
                    )
                    self.runtime.record_run_step(
                        run_id=run_id,
                        step_type="cancelled",
                        status="cancelled",
                        message=reason,
                    )
                    ok("Agent", f"已停止 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={run_id[:8]}")
                except Exception as e:
                    update_langfuse_observation(
                        observation,
                        level="ERROR",
                        status_message=str(e),
                    )
                    self.runtime.record_run_complete(
                        run_id,
                        "failed",
                        partial_response_holder["text"] or final_response_holder["text"],
                        str(e),
                    )
                    await self._emit_stream_event(
                        event_queue,
                        "error",
                        run_id,
                        str(e),
                        status="failed",
                    )
                finally:
                    flush_langfuse()
                    await event_queue.put(sentinel)

        task = asyncio.create_task(run_workflow())
        self._mark_run_active(run_id, task, self.user_id or "", self.session_id or "")
        try:
            while True:
                event = await event_queue.get()
                if event is sentinel:
                    break
                yield event
            await task
        finally:
            if task.done():
                self._mark_run_inactive(run_id)
            else:
                task.add_done_callback(
                    lambda _task, active_run_id=run_id: self._mark_run_inactive(active_run_id)
                )

    async def chat(self, user_input: str) -> str:
        """
        处理用户输入（非流式）
        """
        request_id = str(uuid.uuid4())[:8]

        self._refresh_session_from_store("chat request start")

        # 构建系统提示词
        self.system_prompt = self._build_system_prompt()
        token_message = self._build_context_token_message()
        if token_message:
            info("Memory", f"请求开始 {token_message}")

        # 获取相关历史，仅作为本轮临时上下文传给 LLM
        relevant_history = self._build_relevant_history_context(user_input)

        # 添加到会话：只持久化用户原始输入，避免把检索上下文写入 session.jsonl
        if self.session:
            self.session.add_message("user", user_input)

        # 构建上下文并执行（从 session 获取未整合的历史，包含当前用户输入）
        context_messages = self._get_context_messages()

        graph = self.create_graph()
        initial_state = {
            "messages": context_messages,
            "tool_calls": [],
            "tool_call_chain": [],
            "final_response": "",
            "request_id": request_id,
            "retrieval_context": relevant_history,
        }

        with langfuse_request_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            input_data=user_input,
            metadata={"mode": "sync"},
        ) as observation:
            try:
                result = await graph.ainvoke(initial_state)
                final_response = result["final_response"]
                update_langfuse_observation(
                    observation,
                    output=final_response,
                    metadata=self._langfuse_completion_metadata(request_id, "sync", result),
                )

                # 添加助手回复
                if self.session and final_response:
                    self.session.add_message("assistant", final_response)

                # 保存会话
                if self.session_store and self.session:
                    self.session_store.save_session(self.session)

                # 检查整合
                if self.consolidator and self.session:
                    await self.consolidator.maybe_consolidate(
                        self.session,
                        self.system_prompt,
                        request_id=request_id,
                    )

                return final_response
            except Exception as e:
                update_langfuse_observation(
                    observation,
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            finally:
                flush_langfuse()

    async def chat_eval(self, user_input: str, run_id: str | None = None) -> tuple[str, list[str]]:
        """评测专用轻量链路：不写会话、不归档、不记录 run steps，只执行 Agent 主链路。"""
        request_id = run_id or uuid.uuid4().hex
        self.system_prompt = self._build_system_prompt()
        relevant_history = self._build_relevant_history_context(user_input)
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "tool_calls": [],
            "tool_call_chain": [],
            "final_response": "",
            "request_id": request_id,
            "retrieval_context": relevant_history,
            "event_queue": None,
            "checkpoint_enabled": False,
            "record_steps": False,
        }

        with langfuse_request_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            input_data=user_input,
            metadata={"mode": "eval"},
        ) as observation:
            try:
                result = await self._run_resumable_workflow(initial_state, "llm")
                final_response = str(result.get("final_response") or "")
                tool_calls = [
                    str(item.get("tool") or item.get("name"))
                    for item in list(result.get("tool_call_chain") or [])
                    if isinstance(item, dict) and (item.get("tool") or item.get("name"))
                ]
                update_langfuse_observation(
                    observation,
                    output=final_response,
                    metadata=self._langfuse_completion_metadata(request_id, "eval", result),
                )
                return final_response, tool_calls
            except Exception as e:
                update_langfuse_observation(
                    observation,
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            finally:
                flush_langfuse()

    async def start_new_session(self):
        """兼容旧版“新会话”接口：只清空当前短期会话，不强制归档。"""
        if self.session:
            self.session.messages = []
            self.session.last_consolidated = 0
            self.session.updated_at = datetime.now()
            if self.session_store:
                self.session_store.save_session(self.session)

        ok("Agent", "已清空当前会话，未强制归档")
        return {"archived": 0, "message": "已开始新会话（未强制归档）"}

    async def cleanup(self):
        """清理资源"""
        flush_langfuse()
        if self.consolidator:
            try:
                await self.consolidator.close()
            except Exception as exc:
                warn("Agent", f"关闭记忆整合器失败: {exc}")

        close_fn = getattr(self.llm_client, "close", None) or getattr(self.llm_client, "aclose", None)
        if close_fn:
            try:
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                warn("Agent", f"关闭 LLM 客户端失败: {exc}")

        if self._owns_mcp_client:
            await self.mcp_client.disconnect()
"""
定义了 Agent 的核心逻辑：LLM 推理、工具调用、记忆管理、子代理、支持流式输出
"""

import json
import inspect
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, TypedDict, Sequence
from observability import (
    flush_langfuse,
    get_langfuse_metadata,
    init_langfuse,
    langfuse_request_trace,
    langfuse_tool_span,
    update_langfuse_observation,
)

init_langfuse()
from langfuse.openai import AsyncOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from mcp_client import MCPClient
from config import (
    MODEL_CONFIG,
    SUBAGENT_CONFIG,
    CLARIFICATION_CONFIG,
    AGENT_CONFIG,
    CONCURRENCY_CONFIG,
    LOCAL_TOOL_CONFIG,
)
from industrial_runtime import get_industrial_runtime, sanitize_payload, safe_json
from session_store import SessionStore, SessionData
from memory_manager import VectorHistoryStore
from memory_consolidator import MemoryConsolidator
from skills_loader import SkillsLoader
from tools.registry import ToolRegistry
from tools.shell import ExecTool
from logger import error, info, ok, section, warn
from tools.filesystem import ReadFileTool
from tools.clarification import AskClarificationTool
from subagents import SubAgentRunner, SubAgentConfig, DEFAULT_SUBAGENTS, TASK_TOOL_SYSTEM_PROMPT
from tool_guard import ToolCallGuard
import asyncio
import threading
import uuid

# prompt 文件目录
_PROMPT_DIR = Path(__file__).parent / "prompt"

# 定义状态
class AgentState(TypedDict):
    messages: Sequence[dict]  # 对话历史
    tool_calls: list  # 当前需要调用的工具
    tool_call_chain: list  # 本轮已执行的完整工具调用链
    final_response: str
    request_id: str
    retrieval_context: str
    event_queue: Any


class TravelAgent:
    """Travel Agent - 支持工具调用和长期记忆"""

    _active_run_ids: ClassVar[set[str]] = set()
    _active_run_ids_lock: ClassVar[threading.Lock] = threading.Lock()
    _active_run_tasks: ClassVar[dict[str, asyncio.Task]] = {}
    _active_run_sessions: ClassVar[dict[str, tuple[str, str]]] = {}
    _active_run_tasks_lock: ClassVar[threading.Lock] = threading.Lock()
    _llm_semaphore: ClassVar[asyncio.Semaphore | None] = None
    _llm_semaphore_limit: ClassVar[int] = 0

    def __init__(self, enable_memory: bool = True, session_id: str = None, user_id: str = None, skills_info: str = "",
                 context_window_tokens: int = 9000, max_completion_tokens: int = 1024,
                 workspace: str = None, mcp_client: MCPClient | None = None):
        self.mcp_client = mcp_client or MCPClient()
        self._owns_mcp_client = mcp_client is None
        self.llm_client = AsyncOpenAI(
            api_key=MODEL_CONFIG["api_key"],
            base_url=MODEL_CONFIG["base_url"]
        )
        self.model = MODEL_CONFIG["model_name"]
        self.system_prompt = ""
        self._graph = None
        self.max_workflow_iterations = AGENT_CONFIG["max_iterations"]
        self.tool_guard = ToolCallGuard()
        self._static_prompt_cache_key = None
        self._static_prompt_before_memory = ""
        self._static_prompt_after_memory = ""
        self.session_id = session_id
        self.user_id = user_id or session_id
        self.runtime = get_industrial_runtime()

        # 初始化 SkillsLoader
        self.skills_loader = SkillsLoader()
        self.skills_info = skills_info  # 外部传入的 skill 元数据（优先级更高）

        # 上下文窗口配置
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens

        # 初始化记忆系统
        self.enable_memory = enable_memory
        self.session_store = None
        self.vector_store = None
        self.consolidator = None
        self.session = None  # SessionData 对象

        # 初始化本地工具注册表（用于 skill 执行）
        self.tools = ToolRegistry()
        self._register_default_tools(workspace)

        # 初始化子代理运行器
        self.subagent_runner = None
        if SUBAGENT_CONFIG.get("enabled", True):
            subagent_configs = [
                SubAgentConfig(
                    name=cfg["name"],
                    description=cfg["description"],
                    system_prompt=cfg["system_prompt"],
                    max_iterations=cfg.get("max_iterations", 8),
                )
                for cfg in SUBAGENT_CONFIG.get("subagents", [])
            ] if SUBAGENT_CONFIG.get("subagents") else DEFAULT_SUBAGENTS

            self.subagent_runner = SubAgentRunner(
                llm_client=self.llm_client,
                model=self.model,
                mcp_client=self.mcp_client,
                tools=self.tools,
                subagent_configs=subagent_configs,
            )
            info("Agent", f"子代理已启用 | types={list(self.subagent_runner.subagent_configs.keys())}")

        if enable_memory and self.user_id and session_id:
            try:
                # 初始化会话存储
                self.session_store = SessionStore()
                # 加载或创建会话
                self.session = self.session_store.load_session(self.user_id, session_id)

                # 初始化向量历史存储
                self.vector_store = VectorHistoryStore(self.user_id)
                # 加载并索引现有 HISTORY.md
                history_content = self.session_store.read_history_md(self.user_id)
                if history_content.strip() != "# Conversation History\n\n":
                    self.vector_store.index_history(history_content)

                # 初始化记忆整合器（预算驱动）
                self.consolidator = MemoryConsolidator(
                    session_store=self.session_store,
                    vector_store=self.vector_store,
                    user_id=self.user_id,
                    context_window_tokens=context_window_tokens,
                    max_completion_tokens=max_completion_tokens,
                    safety_buffer=1024
                )

                ok("Agent", f"记忆系统初始化完成 | uid={self.user_id[:8]} sid={session_id[:8]}")
            except Exception as e:
                warn("Agent", f"记忆系统初始化失败: {e}")
                self.enable_memory = False

    def _register_default_tools(self, workspace: str = None) -> None:
        """注册默认的工具集，用于技能执行。"""
        # 设置工作目录
        if workspace is None:
            workspace = os.getcwd()
        workspace_path = Path(workspace)

        # 文件读取仅允许项目目录和 skills 目录，避免公网请求读取主机敏感文件。
        skills_dir = self.skills_loader.skills_dir if hasattr(self.skills_loader, 'skills_dir') else None
        extra_dirs = [skills_dir] if skills_dir else None
        if LOCAL_TOOL_CONFIG.get("read_file_enabled", True):
            self.tools.register(ReadFileTool(
                workspace=workspace_path,
                allowed_dir=workspace_path,
                extra_allowed_dirs=extra_dirs,
            ))

        # 公网部署默认不注册 Shell；只有显式开启时才启用沙箱执行。
        if LOCAL_TOOL_CONFIG.get("shell_enabled", True):
            self.tools.register(ExecTool(
                working_dir=str(workspace_path),
                use_sandbox=True,
                sandbox_config=None  # 自动搜索 .srt-settings.json
            ))

        # 注册 ask_clarification 工具（缺信息时主动向用户澄清）
        if CLARIFICATION_CONFIG.get("enabled", True):
            self.tools.register(AskClarificationTool())

        info("Agent", f"已注册本地工具: {self.tools.tool_names}")

    @classmethod
    def _get_llm_semaphore(cls) -> asyncio.Semaphore | None:
        """按进程共享 LLM 并发阀门，避免高并发评测时把模型服务打爆。"""
        limit = int(CONCURRENCY_CONFIG.get("llm_limit") or 0)
        if limit <= 0:
            return None
        if cls._llm_semaphore is None or cls._llm_semaphore_limit != limit:
            cls._llm_semaphore = asyncio.Semaphore(limit)
            cls._llm_semaphore_limit = limit
        return cls._llm_semaphore

    async def _create_llm_completion(self, **kwargs):
        """统一包一层 LLM 调用，方便按环境变量控制全局并发。"""
        semaphore = self._get_llm_semaphore()
        if semaphore is None:
            return await self.llm_client.chat.completions.create(**kwargs)
        async with semaphore:
            return await self.llm_client.chat.completions.create(**kwargs)

    def _build_system_prompt_uncached(self) -> str:
        """构建系统提示词（参考 nanobot 的 ContextBuilder）"""
        # 获取 MCP 工具描述
        mcp_tools_desc = self.mcp_client.get_tools_description()

        # 获取本地工具描述（用于 skill 执行）
        local_tools_desc = self._get_local_tools_description()

        # 获取 MEMORY.md 内容
        memory_content = ""
        if self.session_store and self.user_id:
            memory_content = self.session_store.read_memory_md(self.user_id)

        # 构建提示词（按照 nanobot 的顺序）
        parts = []

        # 1. 主 agent 提示词（身份 + 工作流程）
        parts.append(self._get_base_system_prompt())

        # 2. MCP 可用工具
        parts.append(f"## MCP 可用工具\n\n{mcp_tools_desc}")

        # 3. 本地工具
        if local_tools_desc:
            parts.append(f"## 本地工具\n\n{local_tools_desc}")

        # 4. task 子代理工具
        if self.subagent_runner:
            task_desc = self.subagent_runner.get_task_tool_description()
            parts.append(f"## 子代理工具\n\n- **task**: {task_desc}")

        # 5. Skill 信息（渐进式加载）
        skills_section = self._build_skills_section()
        if skills_section:
            parts.append(skills_section)

        # 6. 长期记忆（仅当内容非空时）
        if memory_content and memory_content.strip():
            parts.append(f"## 长期记忆\n\n{memory_content}")

        # 7. 子代理使用指南
        if self.subagent_runner:
            subagent_guide = f"{TASK_TOOL_SYSTEM_PROMPT}\n可用子代理类型：\n{self.subagent_runner.get_available_subagents_desc()}"
            parts.append(subagent_guide)

        return "\n\n---\n\n".join(parts)

    def _get_static_system_prompt_parts(self) -> tuple[str, str]:
        cache_key = (
            getattr(self.mcp_client, "tools_version", 0),
            self.skills_info,
            tuple(self.tools.tool_names),
            bool(self.subagent_runner),
        )
        if cache_key == self._static_prompt_cache_key:
            return self._static_prompt_before_memory, self._static_prompt_after_memory

        before_memory = []
        after_memory = []

        mcp_tools_desc = self.mcp_client.get_tools_description()
        local_tools_desc = self._get_local_tools_description()

        before_memory.append(self._get_base_system_prompt())
        before_memory.append(f"## MCP 可用工具\n\n{mcp_tools_desc}")

        if local_tools_desc:
            before_memory.append(f"## 本地工具\n\n{local_tools_desc}")

        if self.subagent_runner:
            task_desc = self.subagent_runner.get_task_tool_description()
            before_memory.append(f"## 子代理工具\n\n- **task**: {task_desc}")

        skills_section = self._build_skills_section()
        if skills_section:
            before_memory.append(skills_section)

        if self.subagent_runner:
            subagent_guide = f"{TASK_TOOL_SYSTEM_PROMPT}\n可用子代理类型：\n{self.subagent_runner.get_available_subagents_desc()}"
            after_memory.append(subagent_guide)

        self._static_prompt_cache_key = cache_key
        self._static_prompt_before_memory = "\n\n---\n\n".join(before_memory)
        self._static_prompt_after_memory = "\n\n---\n\n".join(after_memory)
        return self._static_prompt_before_memory, self._static_prompt_after_memory

    def _build_system_prompt(self) -> str:
        before_memory, after_memory = self._get_static_system_prompt_parts()

        parts = [before_memory]
        memory_content = ""
        if self.session_store and self.user_id:
            memory_content = self.session_store.read_memory_md(self.user_id)

        if memory_content and memory_content.strip():
            parts.append(f"## 长期记忆\n\n{memory_content}")

        if after_memory:
            parts.append(after_memory)

        return "\n\n---\n\n".join(parts)

    def _get_local_tools_description(self) -> str:
        """获取本地工具的描述信息（包含完整参数）"""
        if not self.tools.tool_names:
            return ""

        descriptions = []
        for tool_name in self.tools.tool_names:
            tool = self.tools.get(tool_name)
            if tool:
                desc = f"工具名: {tool_name}\n描述: {tool.description}"
                if tool.parameters and "properties" in tool.parameters:
                    params_desc = []
                    for param_name, param_info in tool.parameters["properties"].items():
                        required = param_name in tool.parameters.get("required", [])
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "无描述")
                        params_desc.append(
                            f"  - {param_name} ({param_type}): {param_desc}{' [必需]' if required else ''}"
                        )
                    if params_desc:
                        desc += f"\n参数:\n{chr(10).join(params_desc)}"
                descriptions.append(desc)
        return "\n\n".join(descriptions)

    _base_system_prompt_cache: ClassVar[str | None] = None

    def _get_base_system_prompt(self) -> str:
        """获取主 agent 完整提示词（身份 + 工作流程，来自 prompt/TRAVEL_AGENT_PROMPT.md）"""
        if TravelAgent._base_system_prompt_cache is None:
            TravelAgent._base_system_prompt_cache = (
                (_PROMPT_DIR / "TRAVEL_AGENT_PROMPT.md")
                .read_text(encoding="utf-8")
                .strip()
            )
        return TravelAgent._base_system_prompt_cache

    def _format_clarification(self, payload: dict, fallback_args: dict) -> str:
        """格式化澄清消息为易读文本。"""
        merged = {
            "question": payload.get("question") or fallback_args.get("question") or "请补充更多信息。",
            "clarification_type": payload.get("clarification_type") or fallback_args.get("clarification_type") or "missing_info",
            "context": payload.get("context") or fallback_args.get("context") or "",
            "options": payload.get("options") or fallback_args.get("options") or [],
        }
        type_icons = {
            "missing_info": "❓",
            "ambiguous_requirement": "❓",
            "approach_choice": "👉",
            "risk_confirmation": "⚠️",
            "suggestion": "💡",
        }
        icon = type_icons.get(merged["clarification_type"], "❓")
        parts: list[str] = []
        if merged["context"]:
            parts.append(f"{icon} {merged['context']}")
            parts.append("")
        parts.append(f"{icon} {merged['question']}")
        if merged["options"]:
            parts.append("")
            for index, option in enumerate(merged["options"], 1):
                parts.append(f"  {index}. {option}")
        return "\n".join(parts)

    def _build_skills_section(self) -> str:
        """构建 Skill 信息部分（参考 nanobot 的 ContextBuilder，渐进式加载）"""
        parts = []

        # 如果外部传入了 skills_info，优先使用
        if self.skills_info:
            parts.append("## 技能信息")
            parts.append(self.skills_info)
            return "\n".join(parts)

        # 否则使用 SkillsLoader 加载
        # 1. 获取所有 skills 的元数据（用于打印）
        all_skills = self.skills_loader.list_skills(filter_unavailable=False)
        if all_skills:
            info("Skills", f"发现 {len(all_skills)} 个 skill")
            for skill in all_skills:
                skill_meta = self.skills_loader._get_skill_meta(skill["name"])
                available = self.skills_loader._check_requirements(skill_meta)
                status = "✓" if available else "✗"
                desc = self.skills_loader._get_skill_description(skill["name"])
                info("Skills", f"  {status} {skill['name']}: {desc[:50]}")

        # 2. 获取 always=true 的 skills 并加载完整内容
        always_skills = self.skills_loader.get_always_skills()
        if always_skills:
            info("Skills", f"自动加载 always=true: {always_skills}")
            always_content = self.skills_loader.load_skills_for_context(always_skills)
            if always_content:
                parts.append("# 激活的技能\n\n这些技能已加载到上下文中，你可以直接使用它们：")
                parts.append(always_content)

        # 3. 获取所有 skills 的元数据摘要
        skills_summary = self.skills_loader.build_skills_summary()
        if skills_summary:
            parts.append("# 可用技能\n")
            parts.append("如需使用某个技能，请使用 read_file 工具读取其 SKILL.md 文件。")
            parts.append("Skills with available=\"false\" 需要先安装依赖。")
            parts.append("")
            parts.append(skills_summary)

        return "\n\n".join(parts) if parts else ""

    def _build_relevant_history_context(self, user_input: str, top_k: int = 2) -> str:
        """构建本轮临时检索上下文；只给 LLM 参考，不写入 session.jsonl。"""
        if not self.vector_store:
            return ""

        results = self.vector_store.search_history(user_input, top_k=top_k)
        if not results:
            return ""

        lines = [
            "## 相关历史记录",
        ]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['full_text']} (相关度: {r['score']:.2f})")

        return "\n".join(lines)

    def _refresh_session_from_store(self, reason: str = "") -> None:
        if not self.session_store or not self.user_id or not self.session_id:
            return

        try:
            old_message_count = len(self.session.messages) if self.session else 0
            old_last_consolidated = (
                self.session.last_consolidated if self.session else 0
            )
            refreshed = self.session_store.load_session(self.user_id, self.session_id)
            self.session = refreshed

            new_message_count = len(refreshed.messages)
            new_last_consolidated = refreshed.last_consolidated
            if (
                new_message_count != old_message_count
                or new_last_consolidated != old_last_consolidated
            ):
                detail = f" | {reason}" if reason else ""
                info(
                    "Memory",
                    "已刷新会话快照"
                    f"{detail} messages={old_message_count}->{new_message_count}"
                    f" last_consolidated={old_last_consolidated}->{new_last_consolidated}",
                )
        except Exception as exc:
            warn("Memory", f"刷新会话快照失败: {exc}")

    def _get_context_messages(self) -> list:
        """获取上下文中使用的消息列表（从 session 中获取未整合的历史消息）"""
        messages = []

        # 添加未整合的历史消息（last_consolidated 之后的消息，包含当前用户输入）
        if self.session:
            unconsolidated = self.session.get_unconsolidated_messages()
            for msg in unconsolidated:
                messages.append({
                    "role": msg["role"],
                    "content": msg.get("content", "")
                })

        return messages

    def _tool_display_name(self, tool_name: str) -> str:
        if tool_name == "task":
            return "子 Agent"
        if tool_name == "ask_clarification":
            return "澄清确认"
        return tool_name

    def _summarize_tool_result(self, result: Any, limit: int = 160) -> str:
        text = result if isinstance(result, str) else safe_json(result, limit=limit * 2)
        text = " ".join(str(text).split())
        if len(text) > limit:
            return text[:limit] + "...(truncated)"
        return text

    def _get_mcp_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """读取 MCP 工具注册时提供的 inputSchema，用于调用前硬校验。"""
        tool = getattr(self.mcp_client, "tools", {}).get(tool_name)
        if not tool:
            return None
        schema = getattr(tool, "inputSchema", None)
        return schema if isinstance(schema, dict) else None

    def _prepare_tool_arguments(self, tool_name: str, tool_args: Any) -> tuple[dict[str, Any], str | None]:
        """执行前校验并尽量复用本地工具已有的参数转换逻辑。"""
        if not isinstance(tool_args, dict):
            return {}, f"工具参数校验失败：parameters must be an object, got {type(tool_args).__name__}"

        if tool_name == "task":
            if not str(tool_args.get("task_description") or "").strip():
                return tool_args, "工具参数校验失败：task_description 参数不能为空。"
            errors = self.tool_guard.validate_arguments(tool_name, tool_args, ToolCallGuard.TASK_SCHEMA)
            return tool_args, self._format_tool_validation_error(tool_name, errors) if errors else None

        if self.tools.has(tool_name):
            _tool, cast_params, error_text = self.tools.prepare_call(tool_name, tool_args)
            if error_text:
                return cast_params, f"工具参数校验失败：{error_text}"
            return cast_params, None

        errors = self.tool_guard.validate_arguments(
            tool_name,
            tool_args,
            self._get_mcp_tool_schema(tool_name),
        )
        return tool_args, self._format_tool_validation_error(tool_name, errors) if errors else None

    @staticmethod
    def _format_tool_validation_error(tool_name: str, errors: list[str]) -> str | None:
        """把 schema 校验错误整理成 LLM 可修正的提示。"""
        if not errors:
            return None
        return (
            f"工具参数校验失败：工具 '{tool_name}' 的参数不符合 schema："
            + "; ".join(errors)
            + "。请根据用户上下文补齐或修正参数后重新调用工具。"
        )

    async def _call_mcp_tool_with_retry(self, tool_name: str, tool_args: dict[str, Any], tool_call_record: dict[str, Any]) -> Any:
        """对 MCP 调用增加外层分类重试，参数类错误不会盲目重试。"""
        attempt = 1
        max_attempts = 1
        while True:
            try:
                tool_call_record["attempt"] = attempt
                tool_call_record["max_attempts"] = max_attempts
                return await self.mcp_client.call_tool(tool_name, tool_args)
            except Exception as exc:
                retry_decision = self.tool_guard.classify_exception(exc)
                max_attempts = retry_decision.max_attempts
                tool_call_record.update({
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error_type": retry_decision.error_type,
                    "retryable": retry_decision.retryable,
                })
                if not retry_decision.retryable or attempt >= max_attempts:
                    raise
                delay = 0.5 * (3 ** (attempt - 1))
                warn(
                    "Tool",
                    f"MCP 工具 {tool_name} 调用失败，{delay:.1f}s 后重试 "
                    f"attempt={attempt}/{max_attempts} type={retry_decision.error_type} err={exc}",
                )
                await asyncio.sleep(delay)
                attempt += 1

    def _langfuse_completion_metadata(self, request_id: str, mode: str, result: dict[str, Any]) -> dict[str, Any]:
        tool_call_chain = list(result.get("tool_call_chain") or [])
        tool_calls = [
            str(item.get("tool") or item.get("name"))
            for item in tool_call_chain
            if isinstance(item, dict) and (item.get("tool") or item.get("name"))
        ]
        return get_langfuse_metadata(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            component="travel_agent",
            operation="chat",
            extra={
                "mode": mode,
                "tool_calls": tool_calls,
                "tool_call_chain": tool_call_chain,
                "tool_call_count": len(tool_calls),
            },
        )

    async def _emit_stream_event(
        self,
        event_queue: asyncio.Queue | None,
        event_type: str,
        run_id: str,
        message: str = "",
        **payload: Any,
    ) -> None:
        record_step = bool(payload.pop("record_step", True))
        event = {
            "type": event_type,
            "run_id": run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            **payload,
        }
        if message:
            event["message"] = message

        if event_type != "content" and record_step:
            status = str(payload.get("status") or "running")
            self.runtime.record_run_step(
                run_id=run_id,
                step_type=event_type,
                status=status,
                message=message,
                tool_name=payload.get("tool"),
                payload=sanitize_payload(payload),
                elapsed_ms=payload.get("elapsed_ms"),
            )

        if event_queue is not None:
            await event_queue.put(event)

    async def _emit_state_event(
        self,
        state: AgentState,
        event_type: str,
        message: str = "",
        **payload: Any,
    ) -> None:
        # 轻量评测路径不需要写 run steps，避免高并发评测把 MySQL 写入放大。
        payload.setdefault("record_step", bool(state.get("record_steps", True)))
        await self._emit_stream_event(
            state.get("event_queue"),
            event_type,
            state.get("request_id", ""),
            message,
            **payload,
        )

    def get_context_token_usage(self) -> tuple[int, int] | None:
        if not self.session:
            return None

        if self.consolidator:
            estimated = self.consolidator.estimate_session_tokens(
                self.session,
                self.system_prompt,
            )
            budget = self.consolidator.budget
        else:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self._get_context_messages())
            estimated = sum(len(str(msg.get("content", ""))) for msg in messages) // 3
            budget = self.context_window_tokens - self.max_completion_tokens - 1024
        return estimated, budget

    def _build_context_token_message(self) -> str | None:
        try:
            usage = self.get_context_token_usage()
            if not usage:
                return None
            estimated, budget = usage
            return f"上下文 token: {estimated}/{budget}"
        except Exception as exc:
            warn("Memory", f"上下文 token 统计失败: {exc}")
            return None

    @classmethod
    def is_run_active(cls, run_id: str) -> bool:
        with cls._active_run_ids_lock:
            return run_id in cls._active_run_ids

    @classmethod
    def _mark_run_active(
        cls,
        run_id: str,
        task: asyncio.Task | None = None,
        user_id: str = "",
        session_id: str = "",
    ) -> None:
        with cls._active_run_ids_lock:
            cls._active_run_ids.add(run_id)
        if task is not None:
            with cls._active_run_tasks_lock:
                cls._active_run_tasks[run_id] = task
                if user_id and session_id:
                    cls._active_run_sessions[run_id] = (user_id, session_id)

    @classmethod
    def _mark_run_inactive(cls, run_id: str) -> None:
        with cls._active_run_ids_lock:
            cls._active_run_ids.discard(run_id)
        with cls._active_run_tasks_lock:
            cls._active_run_tasks.pop(run_id, None)
            cls._active_run_sessions.pop(run_id, None)

    @classmethod
    def cancel_run(cls, run_id: str) -> bool:
        with cls._active_run_tasks_lock:
            task = cls._active_run_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    @classmethod
    def cancel_session_run(cls, user_id: str, session_id: str) -> tuple[str, bool]:
        with cls._active_run_tasks_lock:
            active_items = list(cls._active_run_sessions.items())
        for run_id, (active_user_id, active_session_id) in reversed(active_items):
            if active_user_id == user_id and active_session_id == session_id:
                return run_id, cls.cancel_run(run_id)
        return "", False

    def _raise_if_run_cancelled(self, run_id: str) -> None:
        if self.runtime.is_run_cancelled(run_id):
            raise asyncio.CancelledError("用户主动停止任务")

    def _has_user_message_for_run(self, user_input: str, run_id: str) -> bool:
        if not self.session:
            return False

        for msg in self.session.messages:
            if msg.get("role") == "user" and msg.get("run_id") == run_id:
                return True

        # 兼容崩溃前已经落盘、但旧消息缺 run_id 的情况：只把最后一个未回答用户消息视为本轮输入。
        for msg in reversed(self.session.messages):
            role = msg.get("role")
            if role == "assistant":
                return False
            if role == "user":
                return str(msg.get("content") or "") == user_input
        return False

    def _checkpoint_copy(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return sanitize_payload(value, limit=5000)

    def _checkpoint_payload(self, state: AgentState, next_node: str) -> dict[str, Any]:
        return {
            "version": 1,
            "run_id": state.get("request_id", ""),
            "next_node": next_node,
            "messages": self._checkpoint_copy(list(state.get("messages") or [])),
            "tool_calls": self._checkpoint_copy(list(state.get("tool_calls") or [])),
            "tool_call_chain": self._checkpoint_copy(list(state.get("tool_call_chain") or [])),
            "final_response": str(state.get("final_response") or ""),
            "retrieval_context": str(state.get("retrieval_context") or ""),
        }

    def _save_run_checkpoint(self, state: AgentState, next_node: str) -> None:
        if not state.get("checkpoint_enabled"):
            return
        run_id = str(state.get("request_id") or "")
        if not run_id:
            return
        try:
            self.runtime.record_run_checkpoint(run_id, self._checkpoint_payload(state, next_node))
        except Exception as exc:
            warn("Agent", f"保存任务恢复点失败 | rid={run_id[:8]} err={exc}")

    def _state_from_checkpoint(
        self,
        checkpoint: dict[str, Any] | None,
        event_queue: asyncio.Queue | None,
    ) -> tuple[AgentState | None, str]:
        if not isinstance(checkpoint, dict):
            return None, "llm"
        next_node = str(checkpoint.get("next_node") or "llm")
        if next_node not in {"llm", "tool", "end"}:
            return None, "llm"

        messages = checkpoint.get("messages")
        tool_calls = checkpoint.get("tool_calls")
        if not isinstance(messages, list) or not isinstance(tool_calls, list):
            return None, "llm"

        state: AgentState = {
            "messages": messages,
            "tool_calls": tool_calls,
            "tool_call_chain": list(checkpoint.get("tool_call_chain") or []),
            "final_response": str(checkpoint.get("final_response") or ""),
            "request_id": str(checkpoint.get("run_id") or ""),
            "retrieval_context": str(checkpoint.get("retrieval_context") or ""),
            "event_queue": event_queue,
            "checkpoint_enabled": True,
        }
        if not state["request_id"]:
            return None, "llm"
        return state, next_node

    def _merge_command_update(self, state: AgentState, command: Command) -> AgentState:
        update = getattr(command, "update", None)
        if not isinstance(update, dict):
            return state

        for key, value in update.items():
            if key == "messages":
                messages = list(state.get("messages") or [])
                if isinstance(value, list):
                    messages.extend(value)
                elif value:
                    messages.append(value)
                state["messages"] = messages
            elif key in {"tool_calls", "tool_call_chain", "final_response", "retrieval_context"}:
                state[key] = value
            else:
                state[key] = value
        return state

    def _next_node_from_command(self, state: AgentState, command: Command) -> str:
        goto = getattr(command, "goto", None)
        if goto == END or goto == "__end__":
            return "end"
        if isinstance(goto, str) and goto in {"llm", "tool"}:
            return str(goto)
        if state.get("final_response"):
            return "end"
        return "llm"

    async def _run_resumable_workflow(
        self,
        initial_state: AgentState,
        start_node: str = "llm",
        max_iterations: int | None = None,
    ) -> AgentState:
        state = initial_state
        node = start_node if start_node in {"llm", "tool", "end"} else "llm"
        iteration_limit = max_iterations or self.max_workflow_iterations

        for _ in range(iteration_limit):
            if node == "end":
                self._save_run_checkpoint(state, "end")
                return state

            if node == "llm":
                state = await self.llm_node(state)
                node = self.should_continue(state)
                if node == "end":
                    self._save_run_checkpoint(state, "end")
                    return state
                continue

            if node == "tool":
                tool_result = await self.tool_node(state)
                if hasattr(tool_result, "update") and hasattr(tool_result, "goto"):
                    state = self._merge_command_update(state, tool_result)
                    node = self._next_node_from_command(state, tool_result)
                    self._save_run_checkpoint(state, node)
                    if node == "end":
                        return state
                    continue

                state = tool_result
                node = self._should_continue_after_tool(state)
                if node == "end":
                    self._save_run_checkpoint(state, "end")
                    return state
                continue

        raise RuntimeError(f"Agent 工作流超过最大迭代次数: {iteration_limit}")

    async def llm_node(self, state: AgentState) -> AgentState:
        """
        LLM推理节点
        """
        # 构建完整消息列表
        await self._emit_state_event(state, "status", "正在思考下一步...", status="running")
        system_prompt = self.system_prompt
        retrieval_context = state.get("retrieval_context", "").strip()
        if retrieval_context:
            system_prompt = f"{system_prompt}\n\n---\n\n{retrieval_context}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(state["messages"])

        # 调用 LLM
        response = await self._create_llm_completion(
            model=self.model,
            messages=messages,
            temperature=0.1,
            name="travel-agent-llm",
            metadata=get_langfuse_metadata(
                session_id=self.session_id,
                user_id=self.user_id,
                request_id=state.get("request_id"),
                component="travel_agent",
                operation="llm_node",
            ),
        )
        content = response.choices[0].message.content.strip()
        state["messages"].append({"role": "assistant", "content": content})

        # 解析工具调用
        tool_call = self._parse_tool_call(content)

        info("LLM", f"响应: {content[:120]}")
        info("LLM", f"解析工具调用: {tool_call}")

        if tool_call:
            await self._emit_state_event(
                state,
                "status",
                f"准备调用 {self._tool_display_name(tool_call.get('tool', 'unknown'))} 工具...",
                status="running",
                tool=tool_call.get("tool"),
            )
            state["tool_calls"] = [tool_call]
            state["final_response"] = ""
        else:
            await self._emit_state_event(state, "status", "正在整理最终回答...", status="running")
            state["final_response"] = content
            state["tool_calls"] = []

        self._save_run_checkpoint(
            state,
            "tool" if state.get("tool_calls") else "end",
        )
        return state

    async def tool_node(self, state: AgentState) -> AgentState | Command:
        """
        工具调用节点
        1. 如果是 ask_clarification → 返回 Command(goto=END) 中断本轮
        2. 如果是 task 工具 → 委派给子代理
        3. 否则如果是本地工具（exec, read_file）→ 直接执行
        4. 否则调用 MCP 工具
        """
        if not state.get("tool_calls"):
            return state

        tool_call = state["tool_calls"][0]
        tool_name = tool_call["tool"]
        tool_args = tool_call["arguments"]
        tool_args, validation_error = self._prepare_tool_arguments(tool_name, tool_args)
        display_name = self._tool_display_name(tool_name)
        sanitized_args = sanitize_payload(tool_args)
        tool_call_chain = state.setdefault("tool_call_chain", [])
        signature = self.tool_guard.signature(tool_name, tool_args)

        if validation_error:
            tool_call_record = {
                "tool": tool_name,
                "arguments": sanitized_args,
                "signature": signature,
                "status": "failed",
                "error_type": "validation",
                "retryable": False,
                "error": validation_error,
            }
            tool_call_chain.append(tool_call_record)
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具参数校验失败",
                status="failed",
                tool=tool_name,
                result_summary=validation_error,
            )
            state["messages"].append({"role": "user", "content": validation_error})
            if self.tool_guard.should_stop_after_failure(tool_call_chain):
                state["final_response"] = self.tool_guard.build_failure_stop_response(tool_call_chain)
            state["tool_calls"] = []
            self._save_run_checkpoint(
                state,
                "end" if state.get("final_response") else "llm",
            )
            return state

        guard_decision = self.tool_guard.before_call(tool_call_chain, tool_name, tool_args)
        if guard_decision.action in {"skip", "terminate"}:
            tool_call_record = {
                "tool": tool_name,
                "arguments": sanitized_args,
                "signature": guard_decision.signature,
                "attempt": guard_decision.attempt,
                "status": "blocked" if guard_decision.action == "terminate" else "skipped",
                "result_summary": guard_decision.last_result_summary or guard_decision.message,
            }
            tool_call_chain.append(tool_call_record)
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具重复调用已拦截",
                status=tool_call_record["status"],
                tool=tool_name,
                result_summary=guard_decision.final_response or guard_decision.message,
            )
            if guard_decision.action == "terminate":
                state["final_response"] = guard_decision.final_response
            else:
                state["messages"].append({"role": "user", "content": guard_decision.message})
            state["tool_calls"] = []
            self._save_run_checkpoint(
                state,
                "end" if state.get("final_response") else "llm",
            )
            return state

        tool_call_record = {
            "tool": tool_name,
            "arguments": sanitized_args,
            "signature": signature,
            "attempt": guard_decision.attempt,
            "status": "running",
        }
        tool_call_chain.append(tool_call_record)
        if tool_name == "ask_clarification":
            started_at = time.perf_counter()
            await self._emit_state_event(
                state,
                "tool_start",
                f"正在调用 {display_name} 工具...",
                status="running",
                tool=tool_name,
                arguments=sanitized_args,
            )

        # 拦截 ask_clarification：返回 Command(goto=END) 中断本轮循环
        if tool_name == "ask_clarification":
            info("Tool", "调用澄清工具 ask_clarification")
            try:
                result_str = await self.tools.execute(tool_name, tool_args)
                payload = json.loads(result_str) if isinstance(result_str, str) else {}
            except Exception as exc:
                payload = {}
                tool_call_record.update({"status": "failed", "error": str(exc)})
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            if tool_call_record.get("status") != "failed":
                tool_call_record.update({
                    "status": "success",
                    "elapsed_ms": elapsed_ms,
                    "result_summary": self._summarize_tool_result(payload),
                })
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具已完成",
                status="success",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=self._summarize_tool_result(payload),
            )
            formatted = self._format_clarification(payload, tool_args)
            checkpoint_state = dict(state)
            checkpoint_state["messages"] = list(state.get("messages") or []) + [
                {"role": "assistant", "content": formatted}
            ]
            checkpoint_state["final_response"] = formatted
            checkpoint_state["tool_calls"] = []
            self._save_run_checkpoint(checkpoint_state, "end")
            return Command(
                update={
                    "messages": [{"role": "assistant", "content": formatted}],
                    "final_response": formatted,
                    "tool_calls": [],
                },
                goto=END,
            )

        started_at = time.perf_counter()
        await self._emit_state_event(
            state,
            "tool_start",
            f"正在调用 {display_name} 工具...",
            status="running",
            tool=tool_name,
            arguments=sanitized_args,
        )

        tool_span_cm = langfuse_tool_span(
            tool_name=tool_name,
            arguments=sanitized_args,
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=state.get("request_id"),
        )
        tool_span = tool_span_cm.__enter__()

        try:
            # 1. 检查是否为 task 子代理工具
            if tool_name == "task" and self.subagent_runner:
                info("Tool", f"调用子代理工具 task | 参数={tool_args}")
                subagent_type = tool_args.get("subagent_type", "general-purpose")
                task_description = tool_args.get("task_description", "")
                if not task_description:
                    result = "错误：task_description 参数不能为空"
                else:
                    result = await self.subagent_runner.run(
                        subagent_type,
                        task_description,
                        session_id=self.session_id,
                        user_id=self.user_id,
                        request_id=state.get("request_id"),
                    )
                info("Tool", f"子代理结果: {result[:120]}")

            # 2. 优先检查本地工具（用于 skill 执行）
            elif self.tools.has(tool_name):
                info("Tool", f"调用本地工具 {tool_name} | 参数={tool_args}")
                result = await self.tools.execute(tool_name, tool_args)
                info("Tool", f"结果: {str(result)[:120]}")
            # 3. 否则调用 MCP 工具
            else:
                info("Tool", f"调用 MCP 工具 {tool_name}")
                result = await self._call_mcp_tool_with_retry(tool_name, tool_args, tool_call_record)

            # 工具以 Error/错误 开头时视为失败，避免把失败结果当作有效新增信息继续循环。
            if isinstance(result, str) and result.lstrip().lower().startswith("error"):
                raise RuntimeError(result)
            if isinstance(result, str) and result.lstrip().startswith("错误"):
                raise RuntimeError(result)

            update_langfuse_observation(
                tool_span,
                output=self._summarize_tool_result(result, limit=500),
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            tool_call_record.update({
                "status": "success",
                "elapsed_ms": elapsed_ms,
                "result_summary": self._summarize_tool_result(result),
            })

            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具已返回",
                status="success",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=self._summarize_tool_result(result),
            )

            result_message = (
                f"[工具执行完成]\n"
                f"工具名称: {tool_name}\n"
                f"输入参数: {tool_args}\n"
                f"执行结果: {result}\n\n"
                f"请根据以上结果：\n"
                f"- 如果需要更多信息，调用其他工具\n"
                f"- 如果信息已充分，用中文向用户总结答案"
            )
            state["messages"].append({"role": "user", "content": result_message})
        except RuntimeError as e:
            update_langfuse_observation(tool_span, level="ERROR", status_message=str(e))
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            tool_call_record.update({
                "status": "failed",
                "elapsed_ms": elapsed_ms,
                "error": str(e),
            })
            if "重连失败" in str(e) or "连接异常" in str(e):
                error_message = (
                    f"工具'{tool_name}'暂时不可用（连接已断开且重连失败）。\n"
                    f"请尝试：\n"
                    f"1. 使用其他可用工具完成任务\n"
                    f"2. 或向用户说明情况并建议稍后重试"
                )
            else:
                error_message = f"工具'{tool_name}'执行失败: {str(e)}\n请尝试其他方法或向用户说明。"
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具调用失败",
                status="failed",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=str(e),
            )
            state["messages"].append({"role": "user", "content": error_message})
            if self.tool_guard.should_stop_after_failure(tool_call_chain):
                state["final_response"] = self.tool_guard.build_failure_stop_response(tool_call_chain)
        except Exception as e:
            update_langfuse_observation(tool_span, level="ERROR", status_message=str(e))
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            tool_call_record.update({
                "status": "failed",
                "elapsed_ms": elapsed_ms,
                "error": str(e),
            })
            error_message = f"工具'{tool_name}'执行失败: {str(e)}\n请尝试其他方法或向用户说明。"
            await self._emit_state_event(
                state,
                "tool_result",
                f"{display_name} 工具调用失败",
                status="failed",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                result_summary=str(e),
            )
            state["messages"].append({"role": "user", "content": error_message})
            if self.tool_guard.should_stop_after_failure(tool_call_chain):
                state["final_response"] = self.tool_guard.build_failure_stop_response(tool_call_chain)
        finally:
            try:
                tool_span_cm.__exit__(None, None, None)
            except Exception:
                pass

        state["tool_calls"] = []
        self._save_run_checkpoint(state, "end" if state.get("final_response") else "llm")
        return state

    def should_continue(self, state: AgentState) -> str:
        """
        条件判断函数
        """
        if state.get("tool_calls"):
            return "tool"
        elif state.get("final_response"):
            return "end"
        else:
            return "llm"

    def _normalize_tool_call(self, data: Any) -> dict | None:
        """把解析出的 JSON 数据规范化为单个工具调用。"""
        if isinstance(data, dict) and "tool" in data and "arguments" in data:
            if isinstance(data.get("arguments"), dict):
                return data
            return {
                **data,
                "arguments": {},
            }
        if isinstance(data, list):
            for item in data:
                tool_call = self._normalize_tool_call(item)
                if tool_call:
                    return tool_call
        return None

    def _parse_tool_call(self, content: str) -> dict | None:
        """
        解析工具调用
        """
        content = content.strip()
        decoder = json.JSONDecoder()

        # 情况1: 直接是 JSON
        try:
            data = json.loads(content)
            tool_call = self._normalize_tool_call(data)
            if tool_call:
                return tool_call
        except Exception:
            pass

        # 情况2: Markdown 代码块
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                tool_call = self._normalize_tool_call(data)
                if tool_call:
                    return tool_call
            except Exception:
                pass

        # 情况3: 从任意位置扫描 JSON 对象，兼容嵌套 arguments 和一次输出多个 JSON 的情况。
        for match in re.finditer(r"\{", content):
            try:
                data, _ = decoder.raw_decode(content[match.start():])
            except json.JSONDecodeError:
                continue
            tool_call = self._normalize_tool_call(data)
            if tool_call:
                return tool_call

        # 情况4: 查找最外层的大括号
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and start < end:
            try:
                data = json.loads(content[start:end + 1])
                tool_call = self._normalize_tool_call(data)
                if tool_call:
                    return tool_call
            except Exception:
                pass

        return None

    def create_graph(self):
        """
        创建 LangGraph 工作流
        """
        if self._graph is not None:
            return self._graph

        workflow = StateGraph(AgentState)
        workflow.add_node("llm", self.llm_node)
        workflow.add_node("tool", self.tool_node)
        workflow.add_edge(START, "llm")
        workflow.add_conditional_edges(
            "llm",
            self.should_continue,
            {"tool": "tool", "end": END, "llm": "llm"}
        )
        workflow.add_conditional_edges(
            "tool",
            self._should_continue_after_tool,
            {"end": END, "llm": "llm"}
        )

        self._graph = workflow.compile()
        return self._graph

    def _should_continue_after_tool(self, state: AgentState) -> str:
        """
        工具执行后的条件判断
        - 如果 final_response 已设置（如 ask_clarification），则结束
        - 否则继续 LLM 推理
        """
        if state.get("final_response"):
            return "end"
        else:
            return "llm"

    async def chat_stream(self, user_input: str):
        """
        流式处理用户输入（新版）
        """
        request_id = str(uuid.uuid4())[:8]
        info("Agent", f"开始处理 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={request_id}")

        self._refresh_session_from_store("stream request start")

        # 1. 构建系统提示词
        self.system_prompt = self._build_system_prompt()
        token_message = self._build_context_token_message()
        if token_message:
            info("Memory", f"请求开始 {token_message}")

        # 2. 获取相关历史（向量搜索），仅作为本轮临时上下文传给 LLM
        relevant_history = self._build_relevant_history_context(user_input)

        # 3. 添加到会话：只持久化用户原始输入，避免把检索上下文写入 session.jsonl
        if self.session:
            self.session.add_message("user", user_input)

        # 4. 构建上下文消息（从 session 获取未整合的历史，包含当前用户输入）
        context_messages = self._get_context_messages()

        # 打印完整上下文
        section(f"完整上下文 | rid={request_id}")
        info("Context", f"[SYSTEM PROMPT] {self.system_prompt[:200]}")
        info("Context", f"[USER MESSAGES] 共 {len(context_messages)} 条:")
        for i, msg in enumerate(context_messages):
            info("Context", f"  {i+1}. [{msg['role']}] {str(msg['content'])[:140]}")
        section("END 上下文")

        # 5. 执行工作流
        graph = self.create_graph()
        initial_state = {
            "messages": context_messages,
            "tool_calls": [],
            "tool_call_chain": [],
            "final_response": "",
            "request_id": request_id,
            "retrieval_context": relevant_history,
        }

        with langfuse_request_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            input_data=user_input,
            metadata={"mode": "stream"},
        ) as observation:
            try:
                info("Agent", f"执行工作流 | rid={request_id}")
                result = await graph.ainvoke(initial_state)
                ok("Agent", f"工作流完成 | rid={request_id}")

                final_response = result["final_response"]
                update_langfuse_observation(
                    observation,
                    output=final_response,
                    metadata=self._langfuse_completion_metadata(request_id, "stream", result),
                )

                # 7. 添加助手回复到会话
                if self.session and final_response:
                    self.session.add_message("assistant", final_response)

                # 8. 保存会话到 JSONL
                if self.session_store and self.session:
                    self.session_store.save_session(self.session)
                    ok("Agent", f"已保存会话 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={request_id}")

                # 9. 流式输出
                if final_response:
                    for char in final_response:
                        yield char
                        await asyncio.sleep(0.02)

                # 10. 检查并执行记忆整合
                if self.consolidator and self.session:
                    await self.consolidator.maybe_consolidate(
                        self.session,
                        self.system_prompt,
                        request_id=request_id,
                    )
            except Exception as e:
                update_langfuse_observation(
                    observation,
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            finally:
                flush_langfuse()

        ok("Agent", f"完成 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={request_id}")

    async def chat_stream(
        self,
        user_input: str,
        run_id: str | None = None,
        resume: bool = False,
        content_delay: float | None = None,
    ):
        """流式输出当前用户请求的结构化运行事件。"""
        run_id = run_id or uuid.uuid4().hex
        # 非流式 /chat 可传 0 跳过逐字等待，前端流式默认保留打字机节奏。
        output_delay = (
            float(CONCURRENCY_CONFIG["stream_content_delay"])
            if content_delay is None
            else max(0.0, float(content_delay))
        )
        event_queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()
        final_response_holder = {"text": ""}
        partial_response_holder = {"text": ""}

        async def run_workflow() -> None:
            info("Agent", f"开始处理 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={run_id[:8]}")
            self.runtime.record_run_start(
                run_id=run_id,
                user_id=self.user_id or "",
                session_id=self.session_id or "",
                input_text=user_input,
            )

            with langfuse_request_trace(
                session_id=self.session_id,
                user_id=self.user_id,
                request_id=run_id,
                input_data=user_input,
                metadata={"mode": "stream"},
            ) as observation:
                try:
                    self._raise_if_run_cancelled(run_id)
                    if resume:
                        await self._emit_stream_event(
                            event_queue,
                            "status",
                            run_id,
                            "服务重启后正在接管任务...",
                            status="running",
                        )

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在构建上下文...",
                        status="running",
                    )
                    self._raise_if_run_cancelled(run_id)
                    self._refresh_session_from_store("structured stream request start")
                    self.system_prompt = self._build_system_prompt()
                    token_message = self._build_context_token_message()
                    if token_message:
                        info("Memory", f"请求开始 {token_message}")

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在检索长期记忆...",
                        status="running",
                    )
                    self._raise_if_run_cancelled(run_id)
                    relevant_history = self._build_relevant_history_context(user_input)

                    if self.session:
                        if not self._has_user_message_for_run(user_input, run_id):
                            self.session.add_message("user", user_input, run_id=run_id)
                        if self.session_store:
                            with self.runtime.session_lock(self.user_id or "", self.session_id or ""):
                                self.session_store.save_session(self.session)

                    context_messages = self._get_context_messages()
                    section(f"完整上下文 | rid={run_id[:8]}")
                    info("Context", f"[SYSTEM PROMPT] {self.system_prompt[:200]}")
                    info("Context", f"[USER MESSAGES] 共 {len(context_messages)} 条")
                    for i, msg in enumerate(context_messages):
                        info("Context", f"  {i+1}. [{msg['role']}] {str(msg['content'])[:140]}")
                    section("END 上下文")

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在执行 Agent 工作流...",
                        status="running",
                    )
                    self._raise_if_run_cancelled(run_id)
                    checkpoint_state = None
                    checkpoint_start_node = "llm"
                    if resume:
                        checkpoint_state, checkpoint_start_node = self._state_from_checkpoint(
                            self.runtime.get_latest_run_checkpoint(run_id),
                            event_queue,
                        )
                        if checkpoint_state:
                            await self._emit_stream_event(
                                event_queue,
                                "status",
                                run_id,
                                f"已加载任务恢复点，准备从 {checkpoint_start_node} 继续...",
                                status="running",
                            )

                    initial_state = checkpoint_state or {
                        "messages": context_messages,
                        "tool_calls": [],
                        "tool_call_chain": [],
                        "final_response": "",
                        "request_id": run_id,
                        "retrieval_context": relevant_history,
                        "event_queue": event_queue,
                        "checkpoint_enabled": True,
                    }

                    result = await self._run_resumable_workflow(
                        initial_state,
                        checkpoint_start_node if checkpoint_state else "llm",
                    )
                    self._raise_if_run_cancelled(run_id)
                    final_response = result["final_response"]
                    final_response_holder["text"] = final_response
                    update_langfuse_observation(
                        observation,
                        output=final_response,
                        metadata=self._langfuse_completion_metadata(run_id, "stream", result),
                    )

                    await self._emit_stream_event(
                        event_queue,
                        "status",
                        run_id,
                        "正在输出最终回答...",
                        status="running",
                    )
                    if final_response:
                        for char in final_response:
                            self._raise_if_run_cancelled(run_id)
                            await self._emit_stream_event(
                                event_queue,
                                "content",
                                run_id,
                                content=char,
                            )
                            partial_response_holder["text"] += char
                            if output_delay > 0:
                                await asyncio.sleep(output_delay)

                    self._raise_if_run_cancelled(run_id)
                    if self.session and final_response:
                        self.session.add_message("assistant", final_response, run_id=run_id)

                    if self.session_store and self.session:
                        await self._emit_stream_event(
                            event_queue,
                            "status",
                            run_id,
                            "正在保存会话...",
                            status="running",
                        )
                        self._raise_if_run_cancelled(run_id)
                        with self.runtime.session_lock(self.user_id or "", self.session_id or ""):
                            self.session_store.save_session(self.session)

                    if self.consolidator and self.session:
                        self._raise_if_run_cancelled(run_id)
                        should_consolidate = self.consolidator.should_consolidate(
                            self.session,
                            self.system_prompt,
                        )
                        if should_consolidate:
                            queued = self.runtime.enqueue_task(
                                "memory.consolidate",
                                {
                                    "run_id": run_id,
                                    "user_id": self.user_id,
                                    "session_id": self.session_id,
                                    "message_count": len(self.session.messages),
                                    "last_consolidated": self.session.last_consolidated,
                                    "system_prompt": self.system_prompt,
                                    "context_window_tokens": self.consolidator.context_window_tokens,
                                    "max_completion_tokens": self.consolidator.max_completion_tokens,
                                    "safety_buffer": self.consolidator.safety_buffer,
                                },
                            )
                            if queued:
                                self._raise_if_run_cancelled(run_id)
                                await self._emit_stream_event(
                                    event_queue,
                                    "status",
                                    run_id,
                                    "长期记忆归档已进入后台队列",
                                    status="queued",
                                )
                            else:
                                await self._emit_stream_event(
                                    event_queue,
                                    "status",
                                    run_id,
                                    "正在归档长期记忆...",
                                    status="running",
                                )
                                self._raise_if_run_cancelled(run_id)
                                await self.consolidator.consolidate(
                                    self.session,
                                    self.system_prompt,
                                    request_id=run_id,
                                )

                    self._raise_if_run_cancelled(run_id)
                    token_message = self._build_context_token_message()
                    if token_message:
                        info("Memory", token_message)
                        await self._emit_stream_event(
                            event_queue,
                            "status",
                            run_id,
                            token_message,
                            status="completed",
                        )

                    self.runtime.record_run_complete(run_id, "completed", final_response)
                    await self._emit_stream_event(
                        event_queue,
                        "done",
                        run_id,
                        "本轮任务完成",
                        status="completed",
                        done=True,
                    )
                    ok("Agent", f"完成 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={run_id[:8]}")
                except asyncio.CancelledError as e:
                    reason = str(e) or "用户主动停止任务"
                    update_langfuse_observation(
                        observation,
                        level="WARNING",
                        status_message=reason,
                    )
                    self.runtime.record_run_complete(
                        run_id,
                        "cancelled",
                        partial_response_holder["text"],
                        reason,
                    )
                    self.runtime.record_run_step(
                        run_id=run_id,
                        step_type="cancelled",
                        status="cancelled",
                        message=reason,
                    )
                    ok("Agent", f"已停止 | uid={self.user_id[:8]} sid={self.session_id[:8]} rid={run_id[:8]}")
                except Exception as e:
                    update_langfuse_observation(
                        observation,
                        level="ERROR",
                        status_message=str(e),
                    )
                    self.runtime.record_run_complete(
                        run_id,
                        "failed",
                        partial_response_holder["text"] or final_response_holder["text"],
                        str(e),
                    )
                    await self._emit_stream_event(
                        event_queue,
                        "error",
                        run_id,
                        str(e),
                        status="failed",
                    )
                finally:
                    flush_langfuse()
                    await event_queue.put(sentinel)

        task = asyncio.create_task(run_workflow())
        self._mark_run_active(run_id, task, self.user_id or "", self.session_id or "")
        try:
            while True:
                event = await event_queue.get()
                if event is sentinel:
                    break
                yield event
            await task
        finally:
            if task.done():
                self._mark_run_inactive(run_id)
            else:
                task.add_done_callback(
                    lambda _task, active_run_id=run_id: self._mark_run_inactive(active_run_id)
                )

    async def chat(self, user_input: str) -> str:
        """
        处理用户输入（非流式）
        """
        request_id = str(uuid.uuid4())[:8]

        self._refresh_session_from_store("chat request start")

        # 构建系统提示词
        self.system_prompt = self._build_system_prompt()
        token_message = self._build_context_token_message()
        if token_message:
            info("Memory", f"请求开始 {token_message}")

        # 获取相关历史，仅作为本轮临时上下文传给 LLM
        relevant_history = self._build_relevant_history_context(user_input)

        # 添加到会话：只持久化用户原始输入，避免把检索上下文写入 session.jsonl
        if self.session:
            self.session.add_message("user", user_input)

        # 构建上下文并执行（从 session 获取未整合的历史，包含当前用户输入）
        context_messages = self._get_context_messages()

        graph = self.create_graph()
        initial_state = {
            "messages": context_messages,
            "tool_calls": [],
            "tool_call_chain": [],
            "final_response": "",
            "request_id": request_id,
            "retrieval_context": relevant_history,
        }

        with langfuse_request_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            input_data=user_input,
            metadata={"mode": "sync"},
        ) as observation:
            try:
                result = await graph.ainvoke(initial_state)
                final_response = result["final_response"]
                update_langfuse_observation(
                    observation,
                    output=final_response,
                    metadata=self._langfuse_completion_metadata(request_id, "sync", result),
                )

                # 添加助手回复
                if self.session and final_response:
                    self.session.add_message("assistant", final_response)

                # 保存会话
                if self.session_store and self.session:
                    self.session_store.save_session(self.session)

                # 检查整合
                if self.consolidator and self.session:
                    await self.consolidator.maybe_consolidate(
                        self.session,
                        self.system_prompt,
                        request_id=request_id,
                    )

                return final_response
            except Exception as e:
                update_langfuse_observation(
                    observation,
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            finally:
                flush_langfuse()

    async def chat_eval(self, user_input: str, run_id: str | None = None) -> tuple[str, list[str]]:
        """评测专用轻量链路：不写会话、不归档、不记录 run steps，只执行 Agent 主链路。"""
        request_id = run_id or uuid.uuid4().hex
        self.system_prompt = self._build_system_prompt()
        relevant_history = self._build_relevant_history_context(user_input)
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "tool_calls": [],
            "tool_call_chain": [],
            "final_response": "",
            "request_id": request_id,
            "retrieval_context": relevant_history,
            "event_queue": None,
            "checkpoint_enabled": False,
            "record_steps": False,
        }

        with langfuse_request_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=request_id,
            input_data=user_input,
            metadata={"mode": "eval"},
        ) as observation:
            try:
                result = await self._run_resumable_workflow(initial_state, "llm")
                final_response = str(result.get("final_response") or "")
                tool_calls = [
                    str(item.get("tool") or item.get("name"))
                    for item in list(result.get("tool_call_chain") or [])
                    if isinstance(item, dict) and (item.get("tool") or item.get("name"))
                ]
                update_langfuse_observation(
                    observation,
                    output=final_response,
                    metadata=self._langfuse_completion_metadata(request_id, "eval", result),
                )
                return final_response, tool_calls
            except Exception as e:
                update_langfuse_observation(
                    observation,
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            finally:
                flush_langfuse()

    async def start_new_session(self):
        """兼容旧版“新会话”接口：只清空当前短期会话，不强制归档。"""
        if self.session:
            self.session.messages = []
            self.session.last_consolidated = 0
            self.session.updated_at = datetime.now()
            if self.session_store:
                self.session_store.save_session(self.session)

        ok("Agent", "已清空当前会话，未强制归档")
        return {"archived": 0, "message": "已开始新会话（未强制归档）"}

    async def cleanup(self):
        """清理资源"""
        flush_langfuse()
        if self.consolidator:
            try:
                await self.consolidator.close()
            except Exception as exc:
                warn("Agent", f"关闭记忆整合器失败: {exc}")

        close_fn = getattr(self.llm_client, "close", None) or getattr(self.llm_client, "aclose", None)
        if close_fn:
            try:
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                warn("Agent", f"关闭 LLM 客户端失败: {exc}")

        if self._owns_mcp_client:
            await self.mcp_client.disconnect()
