# mcp_server.py
from fastmcp import FastMCP
from config import MODEL_CONFIG
from observability import flush_langfuse, get_langfuse_metadata, init_langfuse

init_langfuse()
from langfuse.openai import AsyncOpenAI

# 创建 FastMCP 服务器实例
mcp = FastMCP("local-server")
    
llm_client = AsyncOpenAI(
    api_key=MODEL_CONFIG["api_key"],
    base_url=MODEL_CONFIG["base_url"]
)
model_name = MODEL_CONFIG["model_name"]

@mcp.tool()
async def summary_session(conversation_history: str) -> str:
    """调用大模型总结用户和助手的对话历史内容
    Args:
        conversation_history: 包含之前所有的用户和助手的对话历史内容"
    Returns:
        总结后的内容
    """
    try:
        response = await llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的对话总结助手。请简洁地总结用户和助手之间的对话内容，提取关键信息和结论。"},
                {"role": "user", "content": f"请总结以下对话内容：\n\n{conversation_history}"}
            ],
            temperature=0.3,
            name="mcp-summary-session-llm",
            metadata=get_langfuse_metadata(
                component="mcp_server",
                operation="summary_session",
            ),
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"总结失败: {str(e)}"
    finally:
        flush_langfuse()

@mcp.tool()
def add_numbers(a: float, b: float) -> dict:
    """计算两个数字的和
    Args:
        a: 第一个数字
        b: 第二个数字
    Returns:
        包含计算结果的字典
    """
    result = a + b
    return {
        "result": result,
        "message": f"{a} + {b} = {result}"
    }
    
def register_to_app(app, sse_path: str = "/sse", messages_path: str = "/messages/") -> None:
    """将本地 MCP（fastmcp 1.0）的 SSE 端点挂载到现有的 Starlette/FastAPI 应用中，
    避免再启动独立子进程，从根本上节省一次 Python 解释器与依赖（fastmcp、langfuse、openai 等）的冷启动时间。

    挂载后会暴露：
      - GET  {sse_path}        : SSE 长连接入口
      - POST {messages_path}*  : MCP 消息回传端点

    需保证 messages_path 以 "/" 结尾（mcp.server.sse.SseServerTransport 的约束）。
    """
    from mcp.server.sse import SseServerTransport
    from starlette.routing import Mount

    if not messages_path.endswith("/"):
        messages_path = messages_path + "/"

    sse_transport = SseServerTransport(messages_path)

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0],
                streams[1],
                mcp._mcp_server.create_initialization_options(),
            )

    app.add_route(sse_path, handle_sse, methods=["GET"])
    app.router.routes.append(Mount(messages_path, app=sse_transport.handle_post_message))


# 启动服务器（保留独立运行模式以便单独调试，正式部署由 backend.py 内嵌运行）
if __name__ == "__main__":
    from logger import info
    info("MCP", "Server 正在初始化...")

    # FastMCP 默认使用 SSE 传输，默认监听在 http://localhost:8000/sse
    mcp.run(transport="sse")
