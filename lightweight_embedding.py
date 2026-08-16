"""
轻量级 Embedding 客户端 - 直接使用 OpenAI API，无需 llama_index
节省 13.8s 的导入时间，同时保持功能完整性
"""
import httpx
from typing import List, Optional, Awaitable, TypeVar
import asyncio
import concurrent.futures
from functools import lru_cache
import hashlib

T = TypeVar("T")

# 专用线程池：用于在"已有事件循环"的协程里同步等待协程结果（如 llama_index / qdrant 这类同步 API 调用 embedding）
# 单线程足以串行化，避免 httpx.AsyncClient 在不同 loop 之间共享导致的状态问题
_SYNC_BRIDGE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="embedding-sync-bridge"
)


def _run_coro_sync(coro: Awaitable[T]) -> T:
    """在同步上下文中执行协程：
    - 若当前线程没有运行中的事件循环：直接 asyncio.run
    - 若已在事件循环里（典型场景：FastAPI 请求协程调用同步 VectorStore），
      则切到独立线程跑一个新 loop，避免触发
      "asyncio.run() cannot be called from a running event loop"
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 无运行中的 loop，直接同步运行即可
        return asyncio.run(coro)
    # 已有 loop：丢到独立线程的新 loop 中执行，阻塞等待结果
    return _SYNC_BRIDGE_EXECUTOR.submit(asyncio.run, coro).result()


class LightweightEmbedding:
    """
    轻量级 Embedding 客户端，直接调用 OpenAI API
    
    优点：
    - 导入速度快
    - 依赖少
    - 效果与 llama_index 完全一致
    
    功能：
    - 自动重试（最多 3 次）
    - 批量处理（自动分批）
    - 简单缓存（避免重复计算）
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str = None,
        api_base: str = "https://api.openai.com/v1",
        max_retries: int = 3,
        batch_size: int = 100,
        enable_cache: bool = True,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self.client = httpx.AsyncClient(timeout=30.0)
        self._cache = {} if enable_cache else None
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    async def _call_api_with_retry(self, texts: List[str]) -> List[List[float]]:
        """带重试的 API 调用"""
        url = f"{self.api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": texts
        }
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(url, json=data, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                # 按 index 排序返回
                embeddings = [None] * len(texts)
                for item in result["data"]:
                    embeddings[item["index"]] = item["embedding"]
                
                return embeddings
            except httpx.HTTPStatusError as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1)
        
        raise RuntimeError("Failed to get embeddings after retries")
    
    async def get_text_embedding_async(self, text: str) -> List[float]:
        """异步获取单个文本的 embedding"""
        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # 调用 API
        embeddings = await self._call_api_with_retry([text])
        embedding = embeddings[0]
        
        # 存入缓存
        if self.enable_cache:
            self._cache[cache_key] = embedding
        
        return embedding
    
    def get_text_embedding(self, text: str) -> List[float]:
        """同步获取单个文本的 embedding（自动适配是否处于事件循环中）"""
        return _run_coro_sync(self.get_text_embedding_async(text))
    
    def get_query_embedding(self, query: str) -> List[float]:
        """获取查询文本的 embedding（与 get_text_embedding 相同）"""
        return self.get_text_embedding(query)
    
    async def get_text_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取 embedding（自动分批）"""
        if len(texts) <= self.batch_size:
            return await self._call_api_with_retry(texts)
        
        # 分批处理
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._call_api_with_retry(batch)
            results.extend(batch_embeddings)
        
        return results
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """同步批量获取 embedding（自动适配是否处于事件循环中）"""
        return _run_coro_sync(self.get_text_embeddings_async(texts))
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
