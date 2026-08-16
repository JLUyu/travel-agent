#mcp_client.py
"""
本地 MCP 客户端模块 - 负责管理与多个 MCP 服务的连接
通过 SSE 协议使得本地MCP客户端能分别与各个MCP服务端通信
支持并发工具调用、sse连接状态的健康监控、自动重连
"""
import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.sse import sse_client
import httpx
from collections import defaultdict
from config import CONCURRENCY_CONFIG
from logger import error, info, ok, warn

class MCPClient:
    """
    功能:
    1. 并发连接多个 MCP 服务端
    2. 管理工具列表（从各个服务端获取）
    3. 调用工具（支持自动重连）
    4. 监控连接健康状态
    """
    def __init__(self):
        self._stacks: dict[str, AsyncExitStack] = {}
        self._sse_sessions: dict[str, ClientSession] = {}  
        self.tools: dict[str, Tool] = {}
        self._tool_to_sse_session: dict[str, ClientSession] = {}  
        self._sse_session_to_url: dict[ClientSession, str] = {}  
        self._url_tools: dict[str, set[str]] = {}
        self._background_tasks: dict[str, asyncio.Task] = {}
        self._connection_healthy: dict[str, bool] = {}
        self._tools_description_cache: str | None = None
        self._tools_version = 0
        tool_limit = int(CONCURRENCY_CONFIG.get("mcp_tool_limit") or 0)
        # MCP 工具调用限流按进程生效，避免共享 SSE 连接在高并发下被同时写爆。
        self._tool_call_semaphore = (
            asyncio.Semaphore(tool_limit) if tool_limit > 0 else None
        )
        
        # 为每个 URL 创建独立的锁，以便实现并发连接
        self._url_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @property
    def tools_version(self) -> int:
        return self._tools_version

    def _invalidate_tools_cache(self) -> None:
        self._tools_description_cache = None
        self._tools_version += 1

    async def _monitor_connection(self, url: str, sse_session: ClientSession): 
        """
        监控连接健康状态        
        1. 每秒检查一次连接状态
        2. 如果检测到连接断开，标记为不健康
        """
        try:
            while True:
                # 每秒检查一次
                await asyncio.sleep(1)
                
                if hasattr(sse_session, '_read_stream'):
                    if sse_session._read_stream._closed:
                        # 标记连接不健康
                        self._connection_healthy[url] = False
                        warn("MCP", f"检测到连接断开: {url}")
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._connection_healthy[url] = False
            warn("MCP", f"连接监控异常 {url}: {e}")

    async def _connect_internal(self, url: str) -> bool:
        """
        核心连接逻辑
        1. 创建 SSE 客户端连接
        2. 初始化 MCP 会话
        3. 获取服务提供的工具列表
        4. 启动后台监控任务
        """
        try:
            stack = AsyncExitStack()
            sse_cm = sse_client(url.rstrip())
            streams = await stack.enter_async_context(sse_cm)
            sse_session_cm = ClientSession(streams[0], streams[1])
            sse_session = await stack.enter_async_context(sse_session_cm)
            await sse_session.initialize()

            # 获取服务端提供的工具列表
            response = await sse_session.list_tools()
            for old_tool_name in self._url_tools.get(url, set()):
                self.tools.pop(old_tool_name, None)
                self._tool_to_sse_session.pop(old_tool_name, None)

            url_tools = set()
            # 将工具注册到客户端
            for tool in response.tools:
                self.tools[tool.name] = tool
                self._tool_to_sse_session[tool.name] = sse_session
                url_tools.add(tool.name)
            
            # 保存资源栈（便于后续清理资源）
            self._stacks[url] = stack
            self._sse_sessions[url] = sse_session
            self._sse_session_to_url[sse_session] = url
            self._url_tools[url] = url_tools
            self._connection_healthy[url] = True
            self._invalidate_tools_cache()
            
            monitor_task = asyncio.create_task(
                self._monitor_connection(url, sse_session)
            )
            self._background_tasks[url] = monitor_task

            ok("MCP", f"已连接 {url}，加载 {len(response.tools)} 个工具")
            return True
        except Exception as e:
            error_msg = str(e)
            if "localhost" in url or "127.0.0.1" in url:
                warn("MCP", f"连接本地服务失败: {error_msg}")
                warn("MCP", "提示: 请确认 mcp_server.py 已完全启动")
            else:
                warn("MCP", f"连接失败 {url}: {error_msg}")
            return False

    async def connect(self, url: str) -> bool:
        """
        公共连接接口（使用 URL 专属锁）
        不同 URL 的连接可以并发执行，不会互相阻塞
        """
        async with self._url_locks[url]:
            if url in self._sse_sessions and self._connection_healthy.get(url, True):
                return True
            return await self._connect_internal(url)

    async def connect_all(self, urls: list[str]):
        """
        并发连接多个 MCP 服务
        1. 为每个 URL 创建连接任务
        2. 使用 asyncio.gather 并发执行所有任务
        3. 统计成功和失败的数量
        """
        info("MCP", f"并行连接 {len(urls)} 个外部 MCP 服务...")
        
        tasks = [self.connect(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        fail_count = len(urls) - success_count
        
        ok("MCP", f"连接完成: 成功 {success_count}/{len(urls)} 失败 {fail_count}")
        ok("MCP", f"总共加载 {len(self.tools)} 个工具")

    async def _reconnect_sse_session(self, sse_session: ClientSession) -> ClientSession:  
        """重连异常断开的sse连接（使用 URL 专属锁）"""
        url = self._sse_session_to_url.get(sse_session)  
        if not url:
            raise RuntimeError("无法找到 sse_session 对应的 URL")  # 错误信息更新
        
        warn("MCP", f"检测到连接断开，正在重连: {url}")
        
        # 取消旧的监控任务
        old_task = self._background_tasks.pop(url, None)
        if old_task and not old_task.done():
            old_task.cancel()
            try:
                await old_task
            except asyncio.CancelledError:
                pass
        
        # 清理旧连接的资源栈
        old_stack = self._stacks.pop(url, None)
        if old_stack:
            try:
                await asyncio.wait_for(old_stack.aclose(), timeout=2.0)
            except:
                pass
        
        # 调用不加锁的版本（外层已持有 URL 专属锁）
        success = await self._connect_internal(url)
        if not success:
            raise RuntimeError(f"重连失败: {url}")
        
        # 获取新会话
        new_sse_session = self._sse_sessions[url]  
        
        for tool_name, old_sess in list(self._tool_to_sse_session.items()):  
            if old_sess == sse_session:
                self._tool_to_sse_session[tool_name] = new_sse_session  
        
        del self._sse_session_to_url[sse_session]  
        self._sse_session_to_url[new_sse_session] = url  
        
        ok("MCP", f"重连成功: {url}")
        return new_sse_session

    async def call_tool(self, tool_name: str, arguments: dict):
        """按配置限流后调用 MCP 工具。"""
        if self._tool_call_semaphore is None:
            return await self._call_tool_without_limit(tool_name, arguments)
        async with self._tool_call_semaphore:
            return await self._call_tool_without_limit(tool_name, arguments)

    async def _call_tool_without_limit(self, tool_name: str, arguments: dict):
        """
        调用 MCP 工具（支持并发）
        1. 调用前检查连接健康状态，如有问题先重连
        2. 调用失败时自动重连并重试（最多 1 次）
        3. 使用 URL 专属锁，不同 URL 的工具可以并发调用
        """
        if not self.tools:
            raise RuntimeError("MCP客户端未连接")

        info("MCP", f"调用工具 {tool_name} | 参数={arguments}")

        if tool_name not in self._tool_to_sse_session: 
            raise RuntimeError(f"工具 {tool_name} 未找到对应的 sse_session")  
        
        sse_session = self._tool_to_sse_session[tool_name]  
        url = self._sse_session_to_url.get(sse_session)  
        
        # 调用前检查连接健康状态
        if url and not self._connection_healthy.get(url, True):
            warn("MCP", f"连接不健康，尝试先重连: {url}")
            try:
                # 使用 URL 专属锁进行重连（不影响其他 URL）
                async with self._url_locks[url]:
                    sse_session = await self._reconnect_sse_session(sse_session)  
            except Exception as e:
                raise RuntimeError(f"预重连失败: {e}")
        
        # 调用工具（最多重试 1 次）
        for attempt in range(2):
            try:
                # 调用工具（设置 30 秒超时）
                result = await asyncio.wait_for(
                    sse_session.call_tool(tool_name, arguments), 
                    timeout=30.0
                )
                # 将结果处理成str格式
                extracted_result = self._extract_result(result)
                info("MCP", f"工具返回 {tool_name}: {extracted_result[:100]}")
                return extracted_result
                
            except (httpx.ReadError, httpx.RemoteProtocolError, asyncio.TimeoutError) as e:
                if url:
                    self._connection_healthy[url] = False
                
                if attempt == 0:
                    # 第一次失败，尝试重连
                    warn("MCP", f"连接异常 ({type(e).__name__})，尝试重连...")
                    try:
                        # 使用 URL 专属锁进行重连
                        async with self._url_locks[url]:
                            sse_session = await self._reconnect_sse_session(sse_session)  # 调用改名后的方法
                        # 重连成功，继续下一次循环（重试调用工具）
                        continue
                    except Exception as reconnect_error:
                        raise RuntimeError(f"重连失败: {reconnect_error}") from e
                else:
                    raise RuntimeError(f"工具调用失败（已重试）: {e}") from e
            
            except Exception as e:
                error("MCP", f"工具调用异常: {e}")
                import traceback
                traceback.print_exc()
                raise

    def _extract_result(self, result):
        """从工具调用结果中提取文本内容，或将工具结果转为字符串"""
        try:
            # 尝试按照 MCP 协议的标准格式提取
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    first_content = result.content[0]
                    if hasattr(first_content, 'text'):
                        return first_content.text

            # 如果结果本身就是字符串，直接返回
            if isinstance(result, str):
                return result

            # 其他情况，转为字符串
            return str(result)
        except Exception as e:
            warn("MCP", f"提取结果时出错: {e}")
            return str(result)

    def get_tools_description(self) -> str:
        """获取所有工具的描述信息（用于构建 LLM 提示词）"""
        if self._tools_description_cache is not None:
            return self._tools_description_cache

        descriptions = []
        for tool in self.tools.values():
            args_desc = []
            if "properties" in tool.inputSchema:
                for param_name, param_info in tool.inputSchema["properties"].items():
                    required = param_name in tool.inputSchema.get("required", [])
                    args_desc.append(
                        f"  - {param_name} ({param_info.get('type', 'any')}): "
                        f"{param_info.get('description', '无描述')}"
                        f"{' [必需]' if required else ''}"
                    )

            # 构建固定格式的工具描述
            desc = f"工具名: {tool.name}\n描述: {tool.description}"
            # 如果有参数，添加参数描述
            if args_desc:
                desc += f"\n参数:\n{chr(10).join(args_desc)}"
            descriptions.append(desc)
        
        # 用两个换行符连接所有工具描述
        self._tools_description_cache = "\n\n".join(descriptions)
        return self._tools_description_cache

    async def disconnect(self):
        """释放资源"""
        # 1. 取消所有后台任务 
        for url, task in list(self._background_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 2. 关闭所有会话的流
        for url, sse_session in list(self._sse_sessions.items()):  
            try:
                if hasattr(sse_session, '_read_stream'):
                    await sse_session._read_stream.aclose()
                if hasattr(sse_session, '_write_stream'):
                    await sse_session._write_stream.aclose()
            except Exception:
                pass
        
        # 3. 关闭所有资源栈
        for url, stack in list(self._stacks.items()):
            try:
                await asyncio.wait_for(stack.aclose(), timeout=2.0)
            except (asyncio.TimeoutError, RuntimeError, asyncio.CancelledError, Exception):
                pass
        
        # 4. 清空所有数据结构
        self._stacks.clear()
        self._sse_sessions.clear()  
        self.tools.clear()
        self._tool_to_sse_session.clear() 
        self._sse_session_to_url.clear()
        self._url_tools.clear()
        self._background_tasks.clear()
        self._connection_healthy.clear()
        self._invalidate_tools_cache()
        ok("MCP", "客户端已断开")
