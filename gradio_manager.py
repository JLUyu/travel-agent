"""
Gradio 界面管理器

作为前端页面，通过 HTTP 调用后端 FastAPI 服务。
"""

import asyncio
import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque

import aiohttp
import gradio as gr
import requests

from config import PUBLIC_DEPLOYMENT
from logger import info


class MemoryRefreshingChatInterface(gr.ChatInterface):
    """保留项目原有组件名，并使用当前 Gradio 的标准事件实现。"""

    def __init__(
        self,
        *args,
        memory_fn=None,
        memory_inputs=None,
        memory_output=None,
        **kwargs,
    ):
        self.memory_fn = memory_fn
        self.memory_inputs = memory_inputs or []
        self.memory_output = memory_output
        super().__init__(*args, **kwargs)

    def _setup_events(self) -> None:
        super()._setup_events()


class GradioManager:
    """Gradio 前端管理器。"""

    RUN_PROGRESS_MARKER = "<!-- travel-agent-run-progress:"
    RESTORED_STREAM_DELAY = 0.02

    def __init__(self, backend_url: str = "http://127.0.0.1:6008"):
        self.backend_url = backend_url
        self.current_sessions: dict[str, str] = {}
        self.public_deployment = PUBLIC_DEPLOYMENT
        self.rate_limit_per_minute = int(os.getenv("PUBLIC_RATE_LIMIT_PER_MINUTE", "0"))
        self.max_prompt_chars = int(os.getenv("PUBLIC_MAX_PROMPT_CHARS", "8000"))
        self._request_times: dict[str, deque[float]] = defaultdict(deque)
        self._rate_limit_lock = threading.Lock()

    @staticmethod
    def _canonical_uuid(value: str | None) -> str:
        if not value:
            return ""
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, TypeError, AttributeError):
            return ""

    def _allow_request(self, request: gr.Request | None, user_id: str) -> bool:
        """对公网部署做轻量的单实例分钟级限流。"""
        if self.rate_limit_per_minute <= 0:
            return True

        headers = request.headers if request else {}
        forwarded = str(headers.get("x-forwarded-for", "")).split(",", 1)[0].strip()
        connecting_ip = str(headers.get("cf-connecting-ip", "")).strip()
        client = getattr(request, "client", None) if request else None
        client_host = str(getattr(client, "host", "") or "")
        key = connecting_ip or forwarded or client_host or user_id
        now = time.monotonic()

        with self._rate_limit_lock:
            timestamps = self._request_times[key]
            while timestamps and now - timestamps[0] >= 60:
                timestamps.popleft()
            if len(timestamps) >= self.rate_limit_per_minute:
                return False
            timestamps.append(now)
        return True

    def _get_persistent_user_id(self, request: gr.Request | None = None) -> str:
        """
        获取浏览器持久化 user_id。

        首次访问时，如果请求里还没有 cookie，这里会先返回一个兜底 ID。
        正常页面加载流程里，前端 JS 会先写入 cookie，再触发初始化。
        """
        cookies = request.headers.get("cookie", "") if request else ""
        user_id = None

        for cookie in cookies.split(";"):
            cookie = cookie.strip()
            if cookie.startswith("travel_user_id="):
                user_id = cookie.split("=", 1)[1]
                break

        if not user_id:
            user_id = str(uuid.uuid4())
            info("Gradio", f"新用户访问 | uid={user_id[:8]}")
        else:
            info("Gradio", f"老用户返回 | uid={user_id[:8]}")

        return user_id

    def _resolve_user_id(self, user_id: str | None, request: gr.Request | None = None) -> str:
        resolved = user_id or self._get_persistent_user_id(request)
        if not self.public_deployment:
            return resolved
        return self._canonical_uuid(resolved) or str(uuid.uuid4())

    def _get_cookie_value(self, request: gr.Request | None, name: str) -> str:
        cookies = request.headers.get("cookie", "") if request else ""
        for cookie in cookies.split(";"):
            cookie = cookie.strip()
            if cookie.startswith(f"{name}="):
                return cookie.split("=", 1)[1]
        return ""

    def _resolve_session_id(
        self,
        user_id: str,
        session_id: str | None,
        request: gr.Request | None = None,
    ) -> str:
        resolved = session_id or self.current_sessions.get(user_id, "")
        if not self.public_deployment:
            return resolved
        return self._canonical_uuid(resolved)

    def _format_session_choices(self, sessions: list[dict]) -> list[tuple[str, str]]:
        choices = []
        for item in sessions:
            session_id = item["session_id"]
            short = session_id[:8]
            title = item.get("title") or f"会话 {short}"
            label = title if short in title else f"{title} · {short}"
            choices.append((label, session_id))
        return choices

    def _get_sessions(self, user_id: str) -> list[dict]:
        response = requests.get(
            f"{self.backend_url}/session/list/{user_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["sessions"]

    def _create_session(self, user_id: str) -> dict:
        response = requests.post(
            f"{self.backend_url}/session/create/{user_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["session"]

    def _init_session(self, user_id: str, session_id: str):
        response = requests.post(
            f"{self.backend_url}/session/init/{user_id}/{session_id}",
            timeout=30,
        )
        response.raise_for_status()

    def _get_memory_summary(self, user_id: str, session_id: str) -> str:
        response = requests.get(
            f"{self.backend_url}/memory/unconsolidated/{user_id}/{session_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["summary"]

    def _get_messages(self, user_id: str, session_id: str) -> list[dict]:
        response = requests.get(
            f"{self.backend_url}/session/messages/{user_id}/{session_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["messages"]

    def _get_runs(self, user_id: str, session_id: str) -> list[dict]:
        response = requests.get(
            f"{self.backend_url}/session/runs/{user_id}/{session_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("runs", [])

    def _format_run_progress(self, run: dict) -> str:
        lines = []
        for step in run.get("steps", []):
            message = step.get("message") or ""
            if message:
                lines.append(message)
        if not lines:
            lines.append("任务仍在处理中...")
        recent = lines[-8:]
        return "\n".join(f"> {line}" for line in recent)

    def _progress_message(self, run: dict) -> dict:
        status = run.get("status") or "running"
        run_id = run.get("run_id") or ""
        progress = self._format_run_progress(run)
        if status == "failed":
            error_text = run.get("error_text") or "任务失败"
            progress = f"{progress}\n\n任务失败：{error_text}"
        if status == "running":
            marker = f"{self.RUN_PROGRESS_MARKER}{run_id}:{status} -->"
            progress = f"{marker}\n{progress}"
        return {"role": "assistant", "content": progress}

    def _has_progress_message(self, history: list[dict] | None) -> bool:
        for message in reversed(history or []):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "")
            return self.RUN_PROGRESS_MARKER in content
        return False

    def _run_progress_signature(self, run: dict) -> str:
        steps = run.get("steps") or []
        last_step = steps[-1] if steps else {}
        signature_parts = [
            str(run.get("run_id") or ""),
            str(run.get("status") or ""),
            str(len(steps)),
            str(last_step.get("step_id") or last_step.get("id") or ""),
            str(last_step.get("status") or ""),
            str(last_step.get("message") or ""),
            str(len(str(run.get("output_text") or ""))),
        ]
        return "|".join(signature_parts)

    def _runs_progress_signature(self, runs: list[dict]) -> str:
        return "||".join(self._run_progress_signature(run) for run in runs[:3])

    def _runs_memory_signature(self, runs: list[dict]) -> str:
        parts = []
        for run in runs[:3]:
            run_id = str(run.get("run_id") or "")
            for step in run.get("steps") or []:
                if step.get("step_type") != "memory_consolidate":
                    continue
                parts.append(
                    "|".join(
                        [
                            run_id,
                            str(step.get("status") or ""),
                            str(step.get("message") or ""),
                            str(step.get("created_at") or ""),
                        ]
                    )
                )
        return "||".join(parts)

    def _hide_inflight_assistant_message(self, messages: list[dict]) -> list[dict]:
        if messages and messages[-1].get("role") == "assistant":
            return messages[:-1]
        return messages

    def _get_messages_with_progress(self, user_id: str, session_id: str) -> list[dict]:
        messages = self._get_messages(user_id, session_id)
        try:
            runs = self._get_runs(user_id, session_id)
        except Exception as e:
            info("Gradio", f"恢复任务进度失败 | uid={user_id[:8]} sid={session_id[:8]} err={e}")
            return messages

        for run in runs:
            status = run.get("status")
            if status not in {"running", "failed"}:
                continue
            if status == "running":
                messages = self._hide_inflight_assistant_message(messages)
            messages.append(self._progress_message(run))
            break
        return messages

    def _get_memory_summary_update(self, user_id: str, session_id: str):
        try:
            return self._get_memory_summary(user_id, session_id)
        except Exception as e:
            info("Gradio", f"刷新归档情况失败 | uid={user_id[:8]} sid={session_id[:8]} err={e}")
            return gr.update()

    def poll_current_session_progress_signal(
        self,
        user_id: str,
        session_id: str,
        history: list[dict] | None,
        previous_signal: str,
    ):
        """常驻轻量轮询，只在任务进度实际变化时更新隐藏信号。"""
        if not user_id or not session_id:
            return gr.update()

        has_progress = self._has_progress_message(history)
        if not has_progress:
            return gr.update()

        try:
            runs = self._get_runs(user_id, session_id)
        except Exception:
            return gr.update()

        if not runs:
            return gr.update()

        signal = self._runs_progress_signature(runs)
        if signal == previous_signal:
            return gr.update()

        return signal

    def poll_current_session_memory_signal(
        self,
        user_id: str,
        session_id: str,
        previous_signal: str,
    ):
        """轮询后台记忆归档完成信号，只刷新归档情况，不改聊天框。"""
        if not user_id or not session_id:
            return gr.update()

        try:
            runs = self._get_runs(user_id, session_id)
        except Exception:
            return gr.update()

        if any(run.get("status") == "running" for run in runs):
            return gr.update()

        signal = self._runs_memory_signature(runs)
        if not signal or signal == previous_signal:
            return gr.update()

        return signal

    def refresh_current_session_memory(
        self,
        user_id: str,
        session_id: str,
    ):
        if not user_id or not session_id:
            return gr.update()
        return self._get_memory_summary_update(user_id, session_id)

    def render_current_session_progress(
        self,
        user_id: str,
        session_id: str,
        history: list[dict] | None,
    ):
        """按隐藏信号触发实际聊天框刷新，并在完成后补偿式流式输出最终回答。"""
        if not user_id or not session_id:
            yield gr.update(), gr.update()
            return

        has_progress = self._has_progress_message(history)
        if not has_progress:
            yield gr.update(), gr.update()
            return

        try:
            runs = self._get_runs(user_id, session_id)
        except Exception as e:
            yield gr.update(), f"刷新任务进度失败: {str(e)}"
            return

        running_run = next((run for run in runs if run.get("status") == "running"), None)
        if running_run:
            messages = self._hide_inflight_assistant_message(
                self._get_messages(user_id, session_id)
            )
            messages.append(self._progress_message(running_run))
            yield messages, "任务运行中..."
            return

        messages = self._get_messages(user_id, session_id)
        failed_run = next((run for run in runs if run.get("status") == "failed"), None)
        if failed_run:
            messages.append(self._progress_message(failed_run))
            yield messages, "任务失败"
            return

        if not messages or messages[-1].get("role") != "assistant":
            yield messages, "任务已完成"
            return

        final_text = str(messages[-1].get("content") or "")
        prefix = messages[:-1]
        for index in range(1, len(final_text) + 1):
            yield (
                prefix + [{"role": "assistant", "content": final_text[:index]}],
                "正在输出最终回答...",
            )
            time.sleep(self.RESTORED_STREAM_DELAY)
        yield messages, "任务已完成"

    def _cleanup_empty_sessions(self, user_id: str) -> None:
        """请求后端清理该用户所有空会话。失败不抛错，仅记录。"""
        try:
            response = requests.post(
                f"{self.backend_url}/session/cleanup_empty/{user_id}",
                timeout=10,
            )
            response.raise_for_status()
        except Exception as e:
            info("Gradio", f"清理空会话失败 | uid={user_id[:8]} err={e}")

    def switch_session(self, user_id: str, session_id: str, request: gr.Request):
        """切换当前会话。"""
        try:
            user_id = self._resolve_user_id(user_id, request)
            if not session_id:
                return "", "💬 尚未开始对话", [], "请选择会话", ""

            if self.current_sessions.get(user_id) == session_id:
                memory_summary = self._get_memory_summary(user_id, session_id)
                messages = self._get_messages_with_progress(user_id, session_id)
                return session_id, memory_summary, messages, "已就绪", ""

            self._init_session(user_id, session_id)
            self.current_sessions[user_id] = session_id
            memory_summary = self._get_memory_summary(user_id, session_id)
            messages = self._get_messages_with_progress(user_id, session_id)
            return session_id, memory_summary, messages, "已切换会话", ""
        except Exception as e:
            return session_id or "", f"切换失败: {str(e)}", [], "切换失败", ""

    def create_new_session(self, user_id: str, request: gr.Request):
        """创建并切换到一个新会话。"""
        try:
            user_id = self._resolve_user_id(user_id, request)
            session = self._create_session(user_id)
            session_id = session["session_id"]
            self._init_session(user_id, session_id)
            self.current_sessions[user_id] = session_id
            sessions = self._get_sessions(user_id)
            choices = self._format_session_choices(sessions)
            memory_summary = self._get_memory_summary(user_id, session_id)

            return (
                gr.update(choices=choices, value=session_id),
                session_id,
                memory_summary,
                [],
                "已创建新会话",
                "",
            )
        except Exception as e:
            return gr.update(), "", f"创建失败: {str(e)}", [], "创建失败", ""

    def restore_current_session(self, user_id: str, session_id: str):
        """页面刷新后恢复聊天记录和正在运行的中间状态。"""
        if not user_id or not session_id:
            return [], "请选择会话", ""
        try:
            messages = self._get_messages_with_progress(user_id, session_id)
            return messages, "已恢复会话", ""
        except Exception as e:
            return [], f"恢复失败: {str(e)}", ""

    async def chat_function(self, message, history, current_session_id, user_id, request: gr.Request):
        """异步调用后端流式聊天接口。"""
        resolved_user_id = ""
        resolved_session_id = ""
        try:
            user_id = self._resolve_user_id(user_id, request)
            if not self._allow_request(request, user_id):
                yield "❌ 请求过于频繁，请一分钟后再试"
                return
            if len(str(message or "")) > self.max_prompt_chars:
                yield f"❌ 输入过长，请控制在 {self.max_prompt_chars} 个字符以内"
                return
            session_id = self._resolve_session_id(user_id, current_session_id, request)
            resolved_user_id = user_id
            resolved_session_id = session_id
            if not session_id:
                created = self._create_session(user_id)
                session_id = created["session_id"]
                resolved_session_id = session_id
                self.current_sessions[user_id] = session_id

            yield "> 正在连接后端..."

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.backend_url}/chat/stream",
                    json={
                        "message": message,
                        "user_id": user_id,
                        "session_id": session_id,
                    },
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as response:
                    if response.status != 200:
                        yield f"❌ 请求失败: HTTP {response.status}"
                        return

                    full_response = ""
                    status_lines: list[str] = []

                    def render_progress() -> str:
                        if not status_lines:
                            return full_response
                        recent = status_lines[-5:]
                        progress = "\n".join(f"> {line}" for line in recent)
                        if full_response:
                            return f"{progress}\n\n{full_response}"
                        return progress
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue

                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        if "error" in data:
                            yield f"❌ 错误: {data['error']}"
                            return

                        event_type = data.get("type") or ("content" if "content" in data else "")

                        if event_type in {"status", "tool_start", "tool_result"}:
                            message_text = data.get("message", "")
                            if event_type == "tool_start":
                                message_text = message_text or f"正在调用 {data.get('tool', '工具')}..."
                            elif event_type == "tool_result":
                                status = data.get("status", "success")
                                message_text = message_text or f"{data.get('tool', '工具')} 已完成"
                                if status == "failed":
                                    message_text = f"{message_text}: {data.get('result_summary', '')}"
                            if message_text:
                                status_lines.append(message_text)
                                yield render_progress()
                            continue

                        if data.get("done") or event_type == "done":
                            if full_response:
                                yield full_response
                            break

                        if "content" in data:
                            full_response += data["content"]
                            yield render_progress()
        except asyncio.CancelledError:
            if resolved_user_id and resolved_session_id:
                try:
                    await asyncio.to_thread(
                        requests.post,
                        f"{self.backend_url}/runs/cancel-current/{resolved_user_id}/{resolved_session_id}",
                        timeout=5,
                    )
                except Exception as e:
                    info("Gradio", f"停止后端任务失败 | uid={resolved_user_id[:8]} sid={resolved_session_id[:8]} err={e}")
            raise
        except asyncio.TimeoutError:
            yield "❌ 请求超时，请稍后重试"
        except Exception as e:
            yield f"❌ 处理出错: {str(e)}"

    def get_unconsolidated_count(self, current_session_id: str, user_id: str, request: gr.Request):
        """获取当前会话归档情况。"""
        try:
            user_id = self._resolve_user_id(user_id, request)
            session_id = self._resolve_session_id(user_id, current_session_id, request)
            if not session_id:
                return "💬 尚未开始对话"
            return self._get_memory_summary(user_id, session_id)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return "💬 尚未开始对话"
            return f"获取失败: {str(e)}"
        except Exception as e:
            return f"获取失败: {str(e)}"

    def cancel_current_run(self, current_session_id: str, user_id: str, request: gr.Request):
        """停止当前会话正在运行的后端任务。"""
        try:
            user_id = self._resolve_user_id(user_id, request)
            session_id = self._resolve_session_id(user_id, current_session_id, request)
            if not session_id:
                return "当前没有正在运行的任务"
            response = requests.post(
                f"{self.backend_url}/runs/cancel-current/{user_id}/{session_id}",
                timeout=5,
            )
            response.raise_for_status()
            return response.json().get("message", "任务已停止")
        except Exception as e:
            return f"停止失败: {str(e)}"

    def create_interface(self):
        """创建 Gradio 用户界面。"""
        init_user_js = """
        async () => {
            let userId = document.cookie
                .split('; ')
                .find(row => row.startsWith('travel_user_id='))
                ?.split('=')[1];

            if (!userId) {
                userId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                    var r = Math.random() * 16 | 0;
                    var v = c === 'x' ? r : (r & 0x3 | 0x8);
                    return v.toString(16);
                });

                const expires = new Date();
                expires.setTime(expires.getTime() + 365 * 24 * 60 * 60 * 1000);
                document.cookie = `travel_user_id=${userId}; expires=${expires.toUTCString()}; path=/`;
                console.log('生成新用户ID:', userId);
            } else {
                console.log('使用现有用户ID:', userId);
            }

            const prevSessionId = document.cookie
                .split('; ')
                .find(row => row.startsWith('travel_session_id='))
                ?.split('=')[1] || '';

            const emptyDropdown = { choices: [], value: null, __type__: 'update' };

            try {
                await fetch('__BACKEND_URL__/session/cleanup_empty/' + userId + (prevSessionId ? '?exclude=' + prevSessionId : ''), { method: 'POST' });

                let sessionId;

                if (prevSessionId) {
                    const initResp = await fetch('__BACKEND_URL__/session/init/' + userId + '/' + prevSessionId, { method: 'POST' });
                    if (initResp.ok) {
                        sessionId = prevSessionId;
                        console.log('恢复上次会话:', sessionId);
                    }
                }

                if (!sessionId) {
                    const createResp = await fetch('__BACKEND_URL__/session/create/' + userId, { method: 'POST' });
                    if (!createResp.ok) {
                        const errorText = await createResp.text();
                        return [userId, '', `初始化失败: ${errorText || createResp.status}`, emptyDropdown, '初始化失败'];
                    }
                    const createData = await createResp.json();
                    sessionId = createData.session.session_id;

                    const initResp = await fetch('__BACKEND_URL__/session/init/' + userId + '/' + sessionId, { method: 'POST' });
                    if (!initResp.ok) {
                        const errorText = await initResp.text();
                        return [userId, sessionId, `初始化失败: ${errorText || initResp.status}`, emptyDropdown, '初始化失败'];
                    }
                    console.log('首次访问，新建会话:', sessionId);
                }

                const expires = new Date();
                expires.setTime(expires.getTime() + 365 * 24 * 60 * 60 * 1000);
                document.cookie = `travel_session_id=${sessionId}; expires=${expires.toUTCString()}; path=/`;

                const memoryResp = await fetch('__BACKEND_URL__/memory/unconsolidated/' + userId + '/' + sessionId);
                let memorySummary = '💬 尚未开始对话';
                if (memoryResp.ok) {
                    const memoryData = await memoryResp.json();
                    memorySummary = memoryData.summary;
                }

                let dropdownUpdate = emptyDropdown;
                try {
                    const listResp = await fetch('__BACKEND_URL__/session/list/' + userId);
                    if (listResp.ok) {
                        const listData = await listResp.json();
                        const choices = (listData.sessions || []).map(item => {
                            const sid = item.session_id;
                            const short = sid.substring(0, 8);
                            const title = item.title || ('会话 ' + short);
                            const label = title.includes(short) ? title : (title + ' · ' + short);
                            return [label, sid];
                        });
                        dropdownUpdate = { choices: choices, value: sessionId, __type__: 'update' };
                    }
                } catch (e) {
                    console.warn('拉取会话列表失败:', e);
                }

                return [userId, sessionId, memorySummary, dropdownUpdate, '已就绪'];
            } catch (error) {
                console.error('初始化失败:', error);
                return [userId, '', `初始化失败: ${error?.message || error}`, emptyDropdown, '初始化失败'];
            }
        }
        """
        init_user_js = init_user_js.replace("__BACKEND_URL__", self.backend_url)

        persist_session_js = """
        (sessionId) => {
            if (sessionId) {
                const expires = new Date();
                expires.setTime(expires.getTime() + 365 * 24 * 60 * 60 * 1000);
                document.cookie = `travel_session_id=${sessionId}; expires=${expires.toUTCString()}; path=/`;
            }
            return [];
        }
        """

        self.theme = gr.themes.Soft()
        self.css = """
            .gradio-container {
                max-width: 1100px !important;
                margin: 0 auto;
                height: 100vh;
            }
            .sidebar-col {
                border-right: 1px solid var(--border-color-primary);
                padding-right: 16px;
            }
            #main-chatbot {
                height: calc(100vh - 280px) !important;
                min-height: 300px !important;
            }
            #main-chatbot .message.pending[role="status"],
            #main-chatbot .message.pending[aria-label="Loading response"] {
                display: none !important;
            }
            """

        with gr.Blocks(analytics_enabled=False) as demo:
            user_id_state = gr.Textbox(value="", visible=False, show_label=False)
            current_session_id = gr.Textbox(value="", visible=False, show_label=False)
            progress_signal = gr.Textbox(value="", visible=False, show_label=False)
            memory_signal = gr.Textbox(value="", visible=False, show_label=False)
            progress_timer = gr.Timer(value=2.0)

            with gr.Row():
                with gr.Column(scale=1, min_width=220, elem_classes="sidebar-col"):
                    gr.Markdown("# 🧳 Travel Agent")

                    new_session_btn = gr.Button("➕ 新建会话", size="sm")
                    session_dropdown = gr.Dropdown(
                        label="会话",
                        choices=[],
                        interactive=True,
                    )
                    memory_display = gr.Textbox(
                        label="本会话归档情况",
                        value="加载中\n",
                        interactive=False,
                        max_lines=3,
                    )

                    status_output = gr.Textbox(
                        label="操作状态",
                        value="加载中",
                        interactive=False,
                        max_lines=1,
                    )

                with gr.Column(scale=4):
                    custom_chatbot = gr.Chatbot(
                        elem_id="main-chatbot",
                        buttons=[],
                    )
                    chat_interface = MemoryRefreshingChatInterface(
                        fn=self.chat_function,
                        chatbot=custom_chatbot,
                        additional_inputs=[current_session_id, user_id_state],
                        flagging_mode="never",
                        api_visibility="private",
                    )
                    gr.Examples(
                        examples=[
                            "上海坐标多少",
                            "天安门到颐和园怎么走",
                            "2026-06-05 北京到上海火车票有余票吗",
                            "搜索 关于codex 的新闻，并总结第一篇文章",
                            "计算 123+456",
                            "总结对话历史",
                            "使用 skill 获取上海天气",
                            "使用subagent帮我调研上海的景点",
                        ],
                        inputs=chat_interface.textbox,
                    )

            new_session_btn.click(
                fn=self.create_new_session,
                inputs=[user_id_state],
                outputs=[
                    session_dropdown,
                    current_session_id,
                    memory_display,
                    custom_chatbot,
                    status_output,
                    progress_signal,
                ],
                api_visibility="private",
            ).then(
                fn=None,
                inputs=[current_session_id],
                outputs=[],
                js=persist_session_js,
                queue=False,
                api_visibility="private",
            )

            session_dropdown.change(
                fn=self.switch_session,
                inputs=[user_id_state, session_dropdown],
                outputs=[
                    current_session_id,
                    memory_display,
                    custom_chatbot,
                    status_output,
                    progress_signal,
                ],
                queue=False,
                api_visibility="private",
            ).then(
                fn=None,
                inputs=[current_session_id],
                outputs=[],
                js=persist_session_js,
                queue=False,
                api_visibility="private",
            )

            demo.load(
                fn=None,
                inputs=[],
                outputs=[
                    user_id_state,
                    current_session_id,
                    memory_display,
                    session_dropdown,
                    status_output,
                ],
                js=init_user_js,
                queue=False,
                api_visibility="private",
                show_progress="hidden",
            ).then(
                fn=self.restore_current_session,
                inputs=[user_id_state, current_session_id],
                outputs=[custom_chatbot, status_output, progress_signal],
                queue=False,
                api_visibility="private",
                show_progress="hidden",
            )

            progress_timer.tick(
                fn=self.poll_current_session_progress_signal,
                inputs=[user_id_state, current_session_id, custom_chatbot, progress_signal],
                outputs=[progress_signal],
                api_visibility="private",
                show_progress="hidden",
            )

            progress_timer.tick(
                fn=self.poll_current_session_memory_signal,
                inputs=[user_id_state, current_session_id, memory_signal],
                outputs=[memory_signal],
                api_visibility="private",
                show_progress="hidden",
            )

            progress_signal.change(
                fn=self.render_current_session_progress,
                inputs=[user_id_state, current_session_id, custom_chatbot],
                outputs=[custom_chatbot, status_output],
                api_visibility="private",
                show_progress="hidden",
            )

            memory_signal.change(
                fn=self.refresh_current_session_memory,
                inputs=[user_id_state, current_session_id],
                outputs=[memory_display],
                api_visibility="private",
                show_progress="hidden",
            )

        demo.queue(max_size=100, default_concurrency_limit=10)
        return demo
