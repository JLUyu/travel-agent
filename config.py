# config.py

import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

#大模型配置
MODEL_CONFIG = { 
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
}

# 记忆整合需要支持 tool_choice/function calling，不能复用 deepseek-reasoner 这类推理模型
MEMORY_MODEL_CONFIG = {
    "api_key": os.getenv("MEMORY_API_KEY") or MODEL_CONFIG["api_key"],
    "base_url": os.getenv("MEMORY_BASE_URL") or MODEL_CONFIG["base_url"],
    "model_name": os.getenv("MEMORY_MODEL", "deepseek-chat"),
}

# 外部MCP服务器配置
# 高德地图
MCP_CONFIG_AMAP = {"url": os.getenv("AMAP_MCP_URL", "").strip()}

# 火车票查询
MCP_CONFIG_12306 = {"url": os.getenv("RAILWAY_MCP_URL", "").strip()}

# 联网搜索（必应中文搜索）
MCP_CONFIG_WEB_SEARCH = {"url": os.getenv("WEB_SEARCH_MCP_URL", "").strip()}

# 本地 MCP 服务器配置
# 本地 MCP 现已内嵌到 FastAPI 后端进程（端口 6008），无需独立子进程
LOCAL_MCP_CONFIG = {"url": "http://127.0.0.1:6008/sse"}

# 云武配置
EMBEDDING_CONFIG = {
    "api_key": os.getenv("EMBEDDING_API_KEY", ""),
    "model_name": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    "api_base": os.getenv("EMBEDDING_API_BASE", "https://api.wlai.vip/v1"),
}

LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL=os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
LANGFUSE_HOST=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
LANGFUSE_TRACING_ENABLED=os.getenv("LANGFUSE_TRACING_ENABLED", "false")

# 公网部署默认关闭可执行任意命令的 Shell 工具。文件读取始终限制在项目目录内。
PUBLIC_DEPLOYMENT = _env_bool("PUBLIC_DEPLOYMENT")
LOCAL_TOOL_CONFIG = {
    "read_file_enabled": _env_bool("READ_FILE_TOOL_ENABLED", "true"),
    "shell_enabled": _env_bool(
        "SHELL_TOOL_ENABLED",
        "false" if PUBLIC_DEPLOYMENT else "true",
    ),
}

# 子代理配置
SUBAGENT_CONFIG = {
    "enabled": True,  # 是否启用子代理功能
    "max_iterations": 5,  # 子代理默认最大迭代轮次
}

# ask_clarification 功能配置
CLARIFICATION_CONFIG = {
    "enabled": True,  # 是否启用澄清功能（缺信息时主动向用户提问）
}

# 主 Agent 工作流配置
AGENT_CONFIG = {
    # 这里统计的是工作流节点步数（LLM 节点和 tool 节点都会计数），80 大约等于最多 40 次工具调用。
    "max_iterations": int(os.getenv("AGENT_MAX_ITERATIONS", "80")),
}

# 并发配置：0 表示不在应用层限流，交给上游网关或外部服务自身处理。
CONCURRENCY_CONFIG = {
    "llm_limit": int(os.getenv("AGENT_LLM_CONCURRENCY", "0")),
    "mcp_tool_limit": int(os.getenv("AGENT_MCP_TOOL_CONCURRENCY", "0")),
    "stream_content_delay": float(os.getenv("AGENT_STREAM_CONTENT_DELAY", "0.02")),
    "sync_content_delay": float(os.getenv("AGENT_SYNC_CONTENT_DELAY", "0")),
}

# redis、mysql、rabbitmq默认关闭，项目仍可正常运行。
INDUSTRIAL_CONFIG = {
    "mysql": {
        "enabled": _env_bool("MYSQL_ENABLED"),
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "travel_agent"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "travel_agent"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
    },
    "redis": {
        "enabled": _env_bool("REDIS_ENABLED"),
        "url": os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        "ttl_seconds": int(os.getenv("REDIS_TTL_SECONDS", "3600")),
    },
    "rabbitmq": {
        "enabled": _env_bool("RABBITMQ_ENABLED"),
        "url": os.getenv("RABBITMQ_URL", "amqp://guest:guest@127.0.0.1:5672/%2F"),
    },
}
