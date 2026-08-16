"""
记忆整合模块 - 负责将历史对话整合到 MEMORY.md 和 HISTORY.md
参考 nanobot 的实现
"""
import json
import inspect
from datetime import datetime
from typing import List, Dict, Any, Optional

from observability import get_langfuse_metadata, init_langfuse

init_langfuse()
from langfuse.openai import AsyncOpenAI
from config import MEMORY_MODEL_CONFIG
from logger import info, ok, warn


# 定义 save_memory 工具
SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "保存记忆整合结果到持久化存储",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "一段总结关键事件/决策/话题的摘要，以 [YYYY-MM-DD HH:MM] 开头，需要包含尽量多的有用的细节"
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "更新后的长期记忆完整内容（Markdown格式），包含所有现有事实和新事实，如无新内容则返回原内容"
                        "可以写入的内容：用户偏好、实体关系"
                        "请将总长度控制在500字符以内。如果超出，请优先保留最重要的事实。"
                    }
                },
                "required": ["history_entry", "memory_update"]
            }
        }
    }
]


class MemoryConsolidator:
    """
    记忆整合器 - 预算驱动的分段整合
    - 根据上下文窗口预算动态决定整合范围
    - 支持多轮循环整合，直到 prompt 降到目标水位
    - 参考 nanobot 的实现
    """
    
    _MAX_CONSOLIDATION_ROUNDS = 5
    
    def __init__(self, 
                 session_store,
                 vector_store,
                 user_id: str,
                 context_window_tokens: int = 9000,
                 max_completion_tokens: int = 1024,
                 safety_buffer: int = 1024):
        self.session_store = session_store
        self.vector_store = vector_store
        self.user_id = user_id
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.safety_buffer = safety_buffer
        
        # 计算预算和目标
        self._update_budget()
        
        self.llm_client = AsyncOpenAI(
            api_key=MEMORY_MODEL_CONFIG["api_key"],
            base_url=MEMORY_MODEL_CONFIG["base_url"]
        )
        self.model = MEMORY_MODEL_CONFIG["model_name"]
        self._closed = False

    async def close(self) -> None:
        """在事件循环关闭前关闭异步 LLM 客户端。"""
        if self._closed:
            return
        self._closed = True

        close_fn = getattr(self.llm_client, "close", None) or getattr(self.llm_client, "aclose", None)
        if close_fn is None:
            return

        try:
            result = close_fn()
            if inspect.isawaitable(result):
                await result
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc):
                warn("Memory", f"关闭记忆整合客户端时事件循环已关闭: {exc}")
                return
            raise
        except Exception as exc:
            warn("Memory", f"关闭记忆整合客户端失败: {exc}")
    
    def _update_budget(self):
        """计算可用预算和目标水位"""
        self.budget = self.context_window_tokens - self.max_completion_tokens - self.safety_buffer
        self.target = self.budget // 2  # 目标保持在 50% 以下
    
    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        估算消息的 token 数量
        使用简单估算：中文约1.5字符/token，英文约4字符/token
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total_chars += len(content)
        
        # 粗略估算：平均 3 字符/token
        return total_chars // 3
    
    def estimate_session_tokens(self, session, system_prompt: str = "") -> int:
        """
        估算当前会话的 prompt token 数
        包括：系统提示 + 未整合历史消息
        """
        messages = []
        
        # 系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 未整合的历史消息
        unconsolidated = session.get_unconsolidated_messages()
        messages.extend(unconsolidated)
        
        return self.estimate_tokens(messages)
    
    def pick_consolidation_boundary(
        self,
        session,
        tokens_to_remove: int
    ) -> tuple[int, int] | None:
        """
        选择整合边界，确保在用户回合处切割
        返回: (end_index, removed_tokens) 或 None
        """
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None
        
        removed_tokens = 0
        last_boundary = None
        
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            
            # 在用户回合处标记边界（跳过第一条，确保有内容可切）
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            
            # 累加 token（复用 estimate_tokens 的单条逻辑）
            content = message.get("content", "")
            removed_tokens += len(content) // 3
        
        return last_boundary
    
    def should_consolidate(self, session, system_prompt: str = "") -> bool:
        """检查是否需要整合（基于预算）"""
        estimated = self.estimate_session_tokens(session, system_prompt)
        info("Memory", f"上下文 token: {estimated}/{self.budget}")
        return estimated > self.budget
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """格式化消息列表为文本"""
        lines = []
        for msg in messages:
            if not msg.get("content"):
                continue
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "?")[:16]
            lines.append(f"[{timestamp}] {role.upper()}: {content}")
        return "\n".join(lines)
    
    async def _consolidate_chunk(
        self,
        session,
        chunk: list,
        current_memory: str,
        request_id: str | None = None,
    ) -> bool:
        """
        整合指定的消息 chunk
        - 生成历史摘要追加到 HISTORY.md
        - 更新 MEMORY.md
        - 更新向量库
        """
        if not chunk:
            return True
        
        # 构建提示词（只包含 chunk）
        prompt = f"""请处理以下对话片段，并调用 save_memory 工具保存整合结果。

## 当前长期记忆
{current_memory}

## 需要处理的对话片段
{self._format_messages(chunk)}"""

        messages = [
            {"role": "system", "content": "你是记忆整合助手。请调用 save_memory 工具保存对话的整合结果。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # 调用 LLM，强制使用 save_memory 工具
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=SAVE_MEMORY_TOOL,
                tool_choice={"type": "function", "function": {"name": "save_memory"}},
                temperature=0.3,
                name="memory-consolidation-llm",
                metadata=get_langfuse_metadata(
                    session_id=session.session_id,
                    user_id=self.user_id,
                    request_id=request_id,
                    component="memory_consolidator",
                    operation="consolidate_chunk",
                    extra={"chunk_size": len(chunk)},
                ),
            )
            
            # 解析工具调用
            if not response.choices[0].message.tool_calls:
                warn("Memory", "LLM 未调用 save_memory 工具")
                return False
            
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            
            history_entry = args.get("history_entry", "").strip()
            memory_update = args.get("memory_update", "").strip()
            
            if not history_entry:
                warn("Memory", "history_entry 为空")
                return False
            
            # 1. 追加到 HISTORY.md
            self.session_store.append_history_md(self.user_id, history_entry)
            info("Memory", f"已追加历史记录: {history_entry[:50]}")
            
            # 2. 更新向量库
            # 从历史条目中提取时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            if history_entry.startswith("[") and "]" in history_entry:
                timestamp_part = history_entry[1:history_entry.find("]")]
                if len(timestamp_part) == 16:  # YYYY-MM-DD HH:MM
                    timestamp = timestamp_part
            content = history_entry[history_entry.find("]")+1:].strip() if "]" in history_entry else history_entry
            if self.vector_store is not None:
                self.vector_store.add_history_entry(timestamp, content)
            else:
                info("Memory", "后台整合已跳过向量写入：本地 Qdrant 由主服务进程独占")
            
            # 3. 更新 MEMORY.md（如果内容有变化）
            if memory_update and memory_update != current_memory:
                self.session_store.write_memory_md(self.user_id, memory_update)
                ok("Memory", "已更新长期记忆")
            
            return True
            
        except Exception as e:
            warn("Memory", f"记忆整合失败: {e}")
            return False
    
    async def consolidate(
        self,
        session,
        system_prompt: str = "",
        request_id: str | None = None,
    ) -> bool:
        """
        执行记忆整合（预算驱动的多轮循环）
        - 循环整合直到 prompt token 降到目标水位以下
        - 每次只整合超出预算的部分
        """
        if not session.messages:
            return True

        with self.session_store.runtime.memory_lock(self.user_id):
            return await self._consolidate_locked(
                session,
                system_prompt=system_prompt,
                request_id=request_id,
            )

    async def _consolidate_locked(
        self,
        session,
        system_prompt: str = "",
        request_id: str | None = None,
    ) -> bool:
        
        # 读取当前的 MEMORY.md
        current_memory = self.session_store.read_memory_md(self.user_id)
        
        # 多轮循环整合
        for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
            # 估算当前 token
            estimated = self.estimate_session_tokens(session, system_prompt)
            
            # 如果低于目标值，停止整合
            if estimated <= self.target:
                ok("Memory", f"已达到目标水位 {estimated}/{self.target}")
                return True
            
            # 计算需要移除的 token 数
            tokens_to_remove = max(1, estimated - self.target)
            
            # 选择切割边界
            boundary = self.pick_consolidation_boundary(session, tokens_to_remove)
            if boundary is None:
                info("Memory", "无法再找到合适的切割边界，当前整合结束")
                return False
            
            end_idx, removed = boundary
            chunk = session.messages[session.last_consolidated:end_idx]
            
            if not chunk:
                return False
                        
            # 执行整合
            success = await self._consolidate_chunk(
                session,
                chunk,
                current_memory,
                request_id=request_id,
            )
            if not success:
                return False

            # 更新指针并保存
            session.update_last_consolidated(end_idx)
            self.session_store.save_session(session)
            
            current_estimated = self.estimate_session_tokens(session, system_prompt)
            info("Memory", f"第{round_num+1}轮整合完成 | 整合={len(chunk)}条 token={current_estimated}/{self.budget}")

            # 更新 current_memory 用于下一轮
            current_memory = self.session_store.read_memory_md(self.user_id)
        
        warn("Memory", "达到最大整合轮次，结束整合")
        return True

    async def maybe_consolidate(
        self,
        session,
        system_prompt: str = "",
        request_id: str | None = None,
    ) -> bool:
        """
        检查并执行整合（预算驱动的多轮循环）
        """
        if self.should_consolidate(session, system_prompt):
            info("Memory", "Token 超限，触发记忆整合...")
            return await self.consolidate(session, system_prompt, request_id=request_id)
        return False
