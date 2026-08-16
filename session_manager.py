from __future__ import annotations

"""
会话管理器

负责管理多个用户对应的独立 Agent 实例。
所有 Agent 共享 FastAPI 的主事件循环，实现异步并发。
"""

import asyncio
import threading
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from logger import info, ok, warn

if TYPE_CHECKING:
    from travel_agent import TravelAgent


class SessionManager:
    """管理多用户会话对应的 Agent 实例。"""

    def __init__(self):
        self.sessions: dict[tuple[str, str], tuple[Any, datetime]] = {}
        self.lock = threading.Lock()
        self._agent_init_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._mcp_client: Any | None = None
        self._mcp_lock = asyncio.Lock()
        self._mcp_local_ready = False
        self.session_timeout = timedelta(hours=2)

    def create_session(self) -> str:
        """创建新会话并返回 session_id。"""
        return str(uuid.uuid4())

    def _session_key(self, user_id: str, session_id: str) -> tuple[str, str]:
        return (user_id, session_id)

    def _get_agent_init_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        """按会话粒度创建初始化锁，避免不同新会话在冷启动阶段互相排队。"""
        with self.lock:
            lock = self._agent_init_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._agent_init_locks[key] = lock
            return lock

    async def warmup_mcp(self, external_urls: list, local_url: str) -> None:
        try:
            await self._get_or_create_mcp_client(
                external_urls,
                local_url,
                require_local=False,
            )
        except Exception as e:
            warn("Session", f"MCP 预热失败: {e}")

    async def _get_or_create_mcp_client(
        self,
        external_urls: list,
        local_url: str,
        require_local: bool = True,
    ):
        async with self._mcp_lock:
            if self._mcp_client is None:
                from mcp_client import MCPClient

                self._mcp_client = MCPClient()

            await self._mcp_client.connect_all(external_urls)

            if require_local:
                if not self._mcp_local_ready:
                    self._mcp_local_ready = await self._mcp_client.connect(local_url)
                if not self._mcp_local_ready:
                    raise RuntimeError("local MCP server connection failed")

            return self._mcp_client

    async def _initialize_agent(
        self,
        user_id: str,
        session_id: str,
        external_urls: list,
        local_url: str,
        skills_info: str = "",
    ) -> TravelAgent:
        """初始化单个 Agent 实例。"""
        from travel_agent import TravelAgent

        mcp_client = await self._get_or_create_mcp_client(external_urls, local_url)
        return TravelAgent(
            enable_memory=True,
            session_id=session_id,
            user_id=user_id,
            skills_info=skills_info,
            mcp_client=mcp_client,
        )

    async def get_or_create_agent(
        self,
        user_id: str,
        session_id: str,
        external_urls: list,
        local_url: str,
        skills_info: str = "",
    ) -> TravelAgent:
        """获取或创建指定会话的 Agent。"""
        key = self._session_key(user_id, session_id)

        with self.lock:
            if key in self.sessions:
                agent, _ = self.sessions[key]
                self.sessions[key] = (agent, datetime.now())
                return agent

        # 避免同一会话的页面初始化和首条消息并发到达时重复创建 Agent。
        async with self._get_agent_init_lock(key):
            with self.lock:
                if key in self.sessions:
                    agent, _ = self.sessions[key]
                    self.sessions[key] = (agent, datetime.now())
                    return agent

            info("Session", f"为用户 {user_id[:8]} 会话 {session_id[:8]}... 创建新 Agent")
            agent = await self._initialize_agent(
                user_id,
                session_id,
                external_urls,
                local_url,
                skills_info,
            )

            with self.lock:
                self.sessions[key] = (agent, datetime.now())
                self._agent_init_locks.pop(key, None)

            ok("Session", f"Agent 创建成功 | 当前会话数 {len(self.sessions)}")
            return agent

    def get_agent(self, user_id: str, session_id: str) -> TravelAgent:
        """获取指定会话的 Agent。"""
        key = self._session_key(user_id, session_id)
        with self.lock:
            if key not in self.sessions:
                raise ValueError(f"用户 {user_id} 的会话 {session_id} 不存在")

            agent, _ = self.sessions[key]
            self.sessions[key] = (agent, datetime.now())
            return agent

    async def cleanup_session(self, user_id: str, session_id: str):
        """清理指定会话。"""
        key = self._session_key(user_id, session_id)
        with self.lock:
            if key not in self.sessions:
                self._agent_init_locks.pop(key, None)
                return

            agent, _ = self.sessions.pop(key)
            self._agent_init_locks.pop(key, None)

        try:
            await agent.cleanup()
        except Exception as e:
            warn("Session", f"清理 Agent 时出错: {e}")

        ok("Session", f"用户 {user_id[:8]} 会话 {session_id[:8]}... 已清理")

    def cleanup_expired_sessions(self):
        """清理超时会话。"""
        now = datetime.now()
        expired: list[tuple[str, str]] = []

        with self.lock:
            for key, (_, last_active) in self.sessions.items():
                if now - last_active > self.session_timeout:
                    expired.append(key)

        for user_id, session_id in expired:
            asyncio.create_task(self.cleanup_session(user_id, session_id))

        if expired:
            info("Session", f"清理了 {len(expired)} 个超时会话")

    async def cleanup(self):
        """清理所有会话和共享 MCP 连接。"""
        with self.lock:
            agents = [agent for agent, _ in self.sessions.values()]
            self.sessions.clear()
            self._agent_init_locks.clear()

        for agent in agents:
            try:
                await agent.cleanup()
            except Exception as e:
                warn("Session", f"清理 Agent 时出错: {e}")

        if self._mcp_client is not None:
            await self._mcp_client.disconnect()
            self._mcp_client = None
            self._mcp_local_ready = False
