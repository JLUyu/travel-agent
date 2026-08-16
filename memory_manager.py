"""
向量历史存储模块 - 负责 HISTORY.md 的向量化和语义检索
"""
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from lightweight_embedding import LightweightEmbedding
from config import EMBEDDING_CONFIG
from logger import info, ok, warn

class VectorHistoryStore:
    """
    向量历史存储器
    - 将 HISTORY.md 的条目向量化存入 Qdrant
    - 提供语义搜索功能
    """
    
    def __init__(self, 
                 user_id: str,
                 api_key: str = EMBEDDING_CONFIG["api_key"],
                 api_base: str = EMBEDDING_CONFIG["api_base"],
                 db_base_path: str = "./memory"):
        self.user_id = user_id
        self.collection_name = f"history_{user_id}"
        self.db_path = Path(db_base_path) / user_id / "qdrant_vector_db"
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化轻量级 embedding 模型
        info("VecStore", f"初始化向量存储 | uid={user_id[:8]}")
        self.embed_model = LightweightEmbedding(
            model="text-embedding-3-large",
            api_key=api_key,
            api_base=api_base,
        )
        
        # 初始化 Qdrant 客户端
        self.client = QdrantClient(path=str(self.db_path))
        
        # 创建集合
        self._init_collection()
        ok("VecStore", "向量存储初始化完成")
    
    def _init_collection(self):
        """初始化 Qdrant 集合"""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=3072,
                        distance=models.Distance.COSINE
                    ),
                )
                ok("VecStore", f"创建新集合: {self.collection_name}")
        except Exception as e:
            warn("VecStore", f"初始化集合时出错: {e}")
    
    def _generate_id(self, text: str, timestamp: str) -> str:
        """生成唯一 ID"""
        return hashlib.md5(f"{text}_{timestamp}".encode()).hexdigest()
    
    def _parse_history_entries(self, history_content: str) -> List[Dict[str, str]]:
        """解析 HISTORY.md 内容，提取条目"""
        entries = []
        lines = history_content.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # 匹配 [YYYY-MM-DD HH:MM] 格式
            match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] (.+)", line)
            if match:
                timestamp = match.group(1)
                content = match.group(2)
                entries.append({
                    "timestamp": timestamp,
                    "content": content,
                    "full_text": line
                })
        
        return entries
    
    def index_history(self, history_content: str):
        """将 HISTORY.md 内容向量化存入数据库（批量 embedding + 跳过已存在条目）"""
        entries = self._parse_history_entries(history_content)
        
        if not entries:
            return

        all_ids = [self._generate_id(e["content"], e["timestamp"]) for e in entries]

        try:
            existing = self.client.retrieve(
                collection_name=self.collection_name,
                ids=all_ids,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids = {str(p.id) for p in existing}
        except Exception:
            existing_ids = set()

        new_entries = [
            (eid, e) for eid, e in zip(all_ids, entries) if eid not in existing_ids
        ]

        if not new_entries:
            ok("VecStore", f"历史记录已是最新（{len(entries)} 条），跳过 embedding")
            return

        texts = [e["full_text"] for _, e in new_entries]
        embeddings = self.embed_model.get_text_embeddings(texts)

        points = [
            models.PointStruct(
                id=eid,
                vector=emb,
                payload={
                    "timestamp": e["timestamp"],
                    "content": e["content"],
                    "full_text": e["full_text"]
                }
            )
            for (eid, e), emb in zip(new_entries, embeddings)
        ]

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            ok("VecStore", f"索引了 {len(points)} 条历史记录（新增）")
        except Exception as e:
            warn("VecStore", f"索引历史记录时出错: {e}")
    
    def add_history_entry(self, timestamp: str, content: str):
        """添加单条历史记录到向量库"""
        full_text = f"[{timestamp}] {content}"
        
        try:
            embedding = self.embed_model.get_text_embedding(full_text)
            point_id = self._generate_id(content, timestamp)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "timestamp": timestamp,
                            "content": content,
                            "full_text": full_text
                        }
                    )
                ]
            )
        except Exception as e:
            warn("VecStore", f"添加历史记录时出错: {e}")
    
    def search_history(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """语义搜索历史记录"""
        try:
            query_embedding = self.embed_model.get_query_embedding(query)
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=0.5
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "timestamp": hit.payload["timestamp"],
                    "content": hit.payload["content"],
                    "full_text": hit.payload["full_text"],
                    "score": hit.score
                })
            
            return results
        except Exception as e:
            warn("VecStore", f"搜索历史记录时出错: {e}")
            return []
    
    def clear(self):
        """清空向量库"""
        try:
            self.client.delete_collection(self.collection_name)
            self._init_collection()
            ok("VecStore", "向量库已清空")
        except Exception as e:
            warn("VecStore", f"清空向量库时出错: {e}")
    
    def get_stats(self) -> str:
        """获取统计信息"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return f"历史向量库中共有 {collection_info.points_count} 条记录"
        except:
            return "历史向量库为空"
