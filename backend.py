"""
FastAPI 后端服务

负责处理前端请求，并调用对应会话的 Agent。
"""

import asyncio
import json
import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic.warnings import UnsupportedFieldAttributeWarning

from config import (
    CONCURRENCY_CONFIG,
    LOCAL_MCP_CONFIG,
    MCP_CONFIG_12306,
    MCP_CONFIG_AMAP,
    MCP_CONFIG_WEB_SEARCH,
)
from industrial_runtime import get_industrial_runtime
from logger import error, info, ok
from mcp_server import register_to_app as register_local_mcp
from session_manager import SessionManager
from session_store import SessionStore

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

session_manager = SessionManager()
session_store = SessionStore()
industrial_runtime = get_industrial_runtime()

EXTERNAL_URLS = [
    MCP_CONFIG_AMAP["url"],
    MCP_CONFIG_12306["url"],
    MCP_CONFIG_WEB_SEARCH["url"],
] 
EXTERNAL_URLS = [url for url in EXTERNAL_URLS if url]
LOCAL_URL = LOCAL_MCP_CONFIG["url"]

active_requests = 0
active_requests_lock = asyncio.Lock()
resume_tasks: dict[str, asyncio.Task] = {}
resume_tasks_lock = asyncio.Lock()


async def _consume_resumed_run(agent, run_id: str, input_text: str) -> None:
    """后台消费续跑事件，前端通过 /session/runs 读取 MySQL 中的步骤。"""
    try:
        async for _event in agent.chat_stream(input_text, run_id=run_id, resume=True):
            pass
    except Exception as exc:
        industrial_runtime.record_run_complete(run_id, "failed", error_text=str(exc))
        error("Backend", f"接管任务失败 | rid={run_id[:8]} err={exc}")
    finally:
        async with resume_tasks_lock:
            resume_tasks.pop(run_id, None)


async def ensure_running_runs_resumed(user_id: str, session_id: str) -> None:
    """main.py 重启后，接管 MySQL 中仍处于 running 的会话任务。"""
    try:
        from travel_agent import TravelAgent

        runs = industrial_runtime.get_recent_runs(user_id, session_id, limit=5)
        # 同一会话复用一个 Agent 实例，逐个接管可以避免多个续跑任务同时改写同一份 session。
        running_runs = [run for run in runs if run.get("status") == "running"][:1]
        if not running_runs:
            return

        async with resume_tasks_lock:
            resumable_runs = []
            for run in running_runs:
                run_id = str(run.get("run_id") or "")
                input_text = str(run.get("input_text") or "")
                if not run_id or not input_text:
                    continue
                session_snapshot = session_store.load_session(user_id, session_id)
                saved_answer = next(
                    (
                        str(msg.get("content") or "")
                        for msg in reversed(session_snapshot.messages)
                        if msg.get("role") == "assistant" and msg.get("run_id") == run_id
                    ),
                    "",
                )
                if saved_answer:
                    industrial_runtime.record_run_complete(run_id, "completed", saved_answer)
                    continue
                task = resume_tasks.get(run_id)
                if task and not task.done():
                    continue
                if TravelAgent.is_run_active(run_id):
                    continue
                resumable_runs.append((run_id, input_text))

            if not resumable_runs:
                return

            agent = await session_manager.get_or_create_agent(
                user_id,
                session_id,
                EXTERNAL_URLS,
                LOCAL_URL,
            )

            for run_id, input_text in resumable_runs:
                info("Backend", f"接管未完成任务 | uid={user_id[:8]} sid={session_id[:8]} rid={run_id[:8]}")
                industrial_runtime.record_run_step(
                    run_id=run_id,
                    step_type="status",
                    status="running",
                    message="服务重启后正在接管任务...",
                )
                resume_tasks[run_id] = asyncio.create_task(
                    _consume_resumed_run(agent, run_id, input_text)
                )
    except Exception as exc:
        error("Backend", f"检查未完成任务失败: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(session_manager.warmup_mcp(EXTERNAL_URLS, LOCAL_URL))
    yield
    await session_manager.cleanup()


app = FastAPI(title="Travel Agent Backend", version="1.0", lifespan=lifespan)
register_local_mcp(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """聊天请求体。"""

    # promptfoo 等评测场景仅传 query；保留 message/user_id/session_id/run_id 兼容旧前端。
    query: Optional[str] = None
    message: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None

    @property
    def user_input(self) -> str:
        return (self.query if self.query is not None else self.message) or ""


class ChatResponse(BaseModel):
    """聊天响应体。"""

    finalAnswer: str
    tool_calls: list[str]
    user_id: str
    session_id: str
    # 兼容历史调用方
    response: str


class SessionInfo(BaseModel):
    """会话索引项。"""

    session_id: str
    title: str
    created_at: str
    updated_at: str


class SessionListResponse(BaseModel):
    """用户会话列表响应体。"""

    sessions: list[SessionInfo]
    current_session_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    """创建会话响应体。"""

    user_id: str
    session: SessionInfo


class SessionMessagesResponse(BaseModel):
    """会话消息响应体。"""

    messages: list[dict]


class SessionRunsResponse(BaseModel):
    """最近的 Agent 运行进度，用于恢复进行中的状态。"""

    runs: list[dict]


class MemoryResponse(BaseModel):
    """历史对话轮数归档情况响应体。"""

    summary: str
    context_tokens: Optional[int] = None
    context_budget: Optional[int] = None


class StatusResponse(BaseModel):
    """通用状态响应体。"""

    status: str
    message: str


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "sessions": len(session_manager.sessions),
        "active_requests": active_requests,
    }


@app.post("/session/create/{user_id}", response_model=SessionCreateResponse)
async def create_user_session(user_id: str):
    """为指定用户创建新会话。"""
    try:
        session_id = session_manager.create_session()
        session = session_store.create_session(user_id, session_id)
        return SessionCreateResponse(user_id=user_id, session=SessionInfo(**session))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@app.get("/session/list/{user_id}", response_model=SessionListResponse)
async def list_user_sessions(user_id: str):
    """列出指定用户的全部会话。"""
    try:
        sessions = session_store.list_sessions(user_id)
        current_session_id = sessions[0]["session_id"] if sessions else None
        return SessionListResponse(
            sessions=[SessionInfo(**item) for item in sessions],
            current_session_id=current_session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@app.get("/session/messages/{user_id}/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(user_id: str, session_id: str):
    """读取指定会话已保存的消息，用于前端切换会话时恢复聊天窗口。"""
    try:
        session = session_store.load_session(user_id, session_id)
        messages = [
            {"role": msg.get("role"), "content": msg.get("content", "")}
            for msg in session.messages
            if msg.get("role") in {"user", "assistant"} and msg.get("content")
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话消息失败: {str(e)}")


@app.get("/session/runs/{user_id}/{session_id}", response_model=SessionRunsResponse)
async def get_session_runs(user_id: str, session_id: str):
    """读取最近/运行中的 Agent 任务步骤，用于刷新后恢复中间状态。"""
    try:
        await ensure_running_runs_resumed(user_id, session_id)
        runs = industrial_runtime.get_recent_runs(user_id, session_id)
        return SessionRunsResponse(runs=runs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务进度失败: {str(e)}")


@app.post("/session/init/{user_id}/{session_id}", response_model=StatusResponse)
async def initialize_user_session(user_id: str, session_id: str):
    """只初始化会话 Agent，不发送首条消息。"""
    try:
        session_store.create_session(user_id, session_id)
        await session_manager.get_or_create_agent(
            user_id,
            session_id,
            EXTERNAL_URLS,
            LOCAL_URL,
        )
        await ensure_running_runs_resumed(user_id, session_id)
        return StatusResponse(status="success", message="会话初始化成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化会话失败: {str(e)}")


@app.post("/session/init/{session_id}", response_model=StatusResponse)
async def initialize_session_legacy(session_id: str):
    """兼容旧客户端：user_id=session_id。"""
    return await initialize_user_session(session_id, session_id)


@app.delete("/session/{user_id}/{session_id}", response_model=StatusResponse)
async def delete_user_session(user_id: str, session_id: str):
    """删除指定会话（含磁盘文件、索引以及内存中的 Agent）。"""
    try:
        await session_manager.cleanup_session(user_id, session_id)
        removed = session_store.delete_session(user_id, session_id)
        if not removed:
            return StatusResponse(status="noop", message="会话不存在或已被删除")
        return StatusResponse(status="success", message="会话已删除")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")


@app.post("/session/cleanup_empty/{user_id}", response_model=StatusResponse)
async def cleanup_empty_user_sessions(user_id: str, exclude: str = ""):
    """删除指定用户名下所有空会话（无任何用户/助手消息的会话），可通过 exclude 排除指定会话。"""
    try:
        sessions = session_store.list_sessions(user_id)
        removed_count = 0
        for item in sessions:
            session_id = item["session_id"]
            if session_id == exclude:
                continue
            if not session_store.is_session_empty(user_id, session_id):
                continue
            await session_manager.cleanup_session(user_id, session_id)
            if session_store.delete_session(user_id, session_id):
                removed_count += 1
        return StatusResponse(
            status="success",
            message=f"已清理 {removed_count} 个空会话",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理空会话失败: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    非流式聊天接口。

    内部复用与 /chat/stream 完全相同的 chat_stream 流程（包含运行轨迹上报、
    checkpoint、可取消、记忆整合后台队列等），但只把 content 事件聚合成最终
    回答一次性返回，便于其他 Agent 直接调用此接口输入query得到输出。
    """

    global active_requests

    async with active_requests_lock:
        active_requests += 1

    try:
        start_time = time.time()
        user_input = request.user_input
        if not user_input:
            raise HTTPException(status_code=400, detail="query 不能为空")

        user_id = request.user_id or request.session_id or session_manager.create_session()
        session_id = request.session_id or session_manager.create_session()
        session_store.create_session(user_id, session_id)
        info(
            "Backend",
            f"收到同步请求 | 并发={active_requests} uid={user_id[:8]} sid={session_id[:8]}",
        )

        agent = await session_manager.get_or_create_agent(
            user_id,
            session_id,
            EXTERNAL_URLS,
            LOCAL_URL,
        )
        ok("Backend", f"Agent 已就绪 | sid={session_id[:8]}")

        # 复用流式接口的内部流程，但同步接口跳过逐字等待，避免评测并发被输出动画拖慢。
        answer_parts: list[str] = []
        tool_calls_invoked: list[str] = []
        async for chunk in agent.chat_stream(
            user_input,
            run_id=request.run_id,
            content_delay=CONCURRENCY_CONFIG["sync_content_delay"],
        ):
            if not isinstance(chunk, dict):
                answer_parts.append(str(chunk))
                continue
            ev_type = chunk.get("type")
            if ev_type == "content":
                answer_parts.append(chunk.get("content", ""))
            elif ev_type == "tool_start":
                tool_name = chunk.get("tool")
                if tool_name:
                    tool_calls_invoked.append(str(tool_name))
            elif ev_type == "error":
                raise HTTPException(
                    status_code=500,
                    detail=chunk.get("message") or "agent error",
                )

        answer = "".join(answer_parts)
        elapsed = time.time() - start_time
        ok(
            "Backend",
            f"同步请求完成 | chars={len(answer)} tools={tool_calls_invoked} "
            f"elapsed={elapsed:.2f}s sid={session_id[:8]}",
        )
        return ChatResponse(
            finalAnswer=answer,
            tool_calls=tool_calls_invoked,
            response=answer,
            user_id=user_id,
            session_id=session_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        error("Backend", f"异常: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        async with active_requests_lock:
            active_requests -= 1
        info("Backend", f"请求结束 | 并发={active_requests}")


@app.post("/chat/eval", response_model=ChatResponse)
async def chat_eval(request: ChatRequest):
    """评测专用接口：不写会话、不写 run steps、不触发记忆归档，降低高并发评测开销。"""

    global active_requests

    async with active_requests_lock:
        active_requests += 1

    try:
        start_time = time.time()
        user_input = request.user_input
        if not user_input:
            raise HTTPException(status_code=400, detail="query 不能为空")

        user_id = request.user_id or request.session_id or session_manager.create_session()
        session_id = request.session_id or session_manager.create_session()
        session_store.create_session(user_id, session_id)
        info(
            "Backend",
            f"收到评测请求 | 并发={active_requests} uid={user_id[:8]} sid={session_id[:8]}",
        )

        agent = await session_manager.get_or_create_agent(
            user_id,
            session_id,
            EXTERNAL_URLS,
            LOCAL_URL,
        )
        answer, tool_calls_invoked = await agent.chat_eval(
            user_input,
            run_id=request.run_id,
        )
        elapsed = time.time() - start_time
        ok(
            "Backend",
            f"评测请求完成 | chars={len(answer)} tools={tool_calls_invoked} "
            f"elapsed={elapsed:.2f}s sid={session_id[:8]}",
        )
        return ChatResponse(
            finalAnswer=answer,
            tool_calls=tool_calls_invoked,
            response=answer,
            user_id=user_id,
            session_id=session_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        error("Backend", f"评测异常: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        async with active_requests_lock:
            active_requests -= 1
        info("Backend", f"评测请求结束 | 并发={active_requests}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口。

    若 session_id 不存在，则自动创建会话和对应 Agent。
    """

    global active_requests

    try:
        async with active_requests_lock:
            active_requests += 1

        start_time = time.time()
        user_id = request.user_id or request.session_id or session_manager.create_session()
        session_id = request.session_id or session_manager.create_session()
        session_store.create_session(user_id, session_id)
        info(
            "Backend",
            f"收到流式请求 | 并发={active_requests} uid={user_id[:8]} sid={session_id[:8]}",
        )

        agent = await session_manager.get_or_create_agent(
            user_id,
            session_id,
            EXTERNAL_URLS,
            LOCAL_URL,
        )
        ok("Backend", f"Agent 已就绪 | sid={session_id[:8]}")

        async def generate():
            global active_requests

            try:
                chunk_count = 0
                done_sent = False
                async for chunk in agent.chat_stream(request.user_input, run_id=request.run_id):
                    if isinstance(chunk, dict):
                        payload = {
                            "user_id": user_id,
                            "session_id": session_id,
                            **chunk,
                        }
                    else:
                        payload = {
                            "type": "content",
                            "content": chunk,
                            "user_id": user_id,
                            "session_id": session_id,
                        }

                    if payload.get("type") == "content":
                        chunk_count += 1
                    if payload.get("type") == "done":
                        done_sent = True
                    if chunk_count == 1:
                        info("Backend", f"开始输出首字符 | sid={session_id[:8]}")

                    yield (
                        "data: "
                        + json.dumps(payload, ensure_ascii=False)
                        + "\n\n"
                    )

                if not done_sent:
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "done",
                                "done": True,
                                "user_id": user_id,
                                "session_id": session_id,
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

                elapsed = time.time() - start_time
                ok(
                    "Backend",
                    f"请求完成 | chars={chunk_count} elapsed={elapsed:.2f}s sid={session_id[:8]}",
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                async with active_requests_lock:
                    active_requests -= 1
                info("Backend", f"请求结束 | 并发={active_requests} sid={session_id[:8]}")

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        async with active_requests_lock:
            active_requests -= 1
        error("Backend", f"异常: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/runs/{run_id}/cancel", response_model=StatusResponse)
async def cancel_run(run_id: str):
    """取消一个正在运行的 Agent 任务。"""
    try:
        from travel_agent import TravelAgent

        industrial_runtime.request_run_cancel(run_id)
        stopped = TravelAgent.cancel_run(run_id)
        async with resume_tasks_lock:
            task = resume_tasks.get(run_id)
            if task and not task.done():
                task.cancel()
                stopped = True
        message = "任务已停止" if stopped else "任务已标记为停止"
        ok("Backend", f"{message} | rid={run_id[:8]}")
        return StatusResponse(status="success", message=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止任务失败: {str(e)}")


@app.post("/runs/cancel-current/{user_id}/{session_id}", response_model=StatusResponse)
async def cancel_current_run(user_id: str, session_id: str):
    """取消当前会话最近的 running Agent 任务。"""
    try:
        from travel_agent import TravelAgent

        runs = industrial_runtime.get_recent_runs(user_id, session_id, limit=5)
        running_run = next((run for run in runs if run.get("status") == "running"), None)
        if running_run:
            return await cancel_run(str(running_run["run_id"]))

        active_run_id, stopped = TravelAgent.cancel_session_run(user_id, session_id)
        if active_run_id:
            industrial_runtime.request_run_cancel(active_run_id)
            message = "任务已停止" if stopped else "任务已标记为停止"
            ok("Backend", f"{message} | rid={active_run_id[:8]}")
            return StatusResponse(status="success", message=message)

        return StatusResponse(status="noop", message="当前没有正在运行的任务")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止当前任务失败: {str(e)}")


@app.get("/memory/unconsolidated/{user_id}/{session_id}", response_model=MemoryResponse)
async def get_user_unconsolidated_count(user_id: str, session_id: str):
    """获取指定会话的记忆统计。"""
    try:
        session = session_store.load_session(user_id, session_id)
        total = len(session.messages)
        consolidated = session.last_consolidated
        unconsolidated = total - consolidated
        summary_lines = [
            f"💬 未归档: {unconsolidated} 条",
            f"💬 已归档: {consolidated} 条",
        ]

        context_tokens = None
        context_budget = None
        try:
            agent = session_manager.get_agent(user_id, session_id)
            agent._refresh_session_from_store("memory summary")
            agent.system_prompt = agent._build_system_prompt()
            usage = agent.get_context_token_usage()
            if usage:
                context_tokens, context_budget = usage
                summary_lines.append(f"🧠 上下文 token: {context_tokens}/{context_budget}")
        except ValueError:
            summary_lines.append("🧠 上下文 token: Agent 未初始化")
        except Exception as e:
            error("Backend", f"上下文 token 统计失败: {e}")
            summary_lines.append("🧠 上下文 token: 统计失败")

        return MemoryResponse(
            summary="\n".join(summary_lines),
            context_tokens=context_tokens,
            context_budget=context_budget,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@app.get("/memory/unconsolidated/{session_id}", response_model=MemoryResponse)
async def get_unconsolidated_count_legacy(session_id: str):
    """兼容旧客户端：user_id=session_id。"""
    return await get_user_unconsolidated_count(session_id, session_id)


@app.post("/session/new/{user_id}/{session_id}", response_model=StatusResponse)
async def start_user_new_session(user_id: str, session_id: str):
    """兼容旧客户端的新会话接口：清空当前短期会话，不触发强制归档。"""
    try:
        agent = session_manager.get_agent(user_id, session_id)
        result = await agent.start_new_session()
        return StatusResponse(status="success", message=result["message"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"开始新会话失败: {str(e)}")


@app.post("/session/new/{session_id}", response_model=StatusResponse)
async def start_new_session_legacy(session_id: str):
    """兼容旧客户端：user_id=session_id。"""
    return await start_user_new_session(session_id, session_id)


if __name__ == "__main__":
    import uvicorn

    workers = max(1, int(os.getenv("BACKEND_WORKERS", "1")))
    # 多 worker 必须使用 import string，方便 uvicorn 为每个进程独立加载 app。
    target = "backend:app" if workers > 1 else app
    backend_port = int(os.getenv("BACKEND_PORT", "6008"))
    uvicorn.run(target, host="0.0.0.0", port=backend_port, log_level="info", workers=workers)
