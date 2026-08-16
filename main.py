#main.py
"""
主启动文件 - 启动所有服务

优化后流程：
1. 本地 MCP 已内嵌进 FastAPI 后端进程（mcp_server.register_to_app），不再单独起子进程
2. 立即启动 FastAPI 后端（不再有任何冗余 sleep）
3. 在等待后端就绪的同时，主进程并行准备 Gradio 前端
"""

import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import subprocess
import sys
import time
import asyncio
import socket
from urllib.parse import urlparse
from observability import flush_langfuse, init_langfuse
from logger import banner, error, info, ok, warn


BACKEND_PORT = int(os.getenv("BACKEND_PORT", "6008"))
BACKEND_URL = os.getenv("BACKEND_URL", f"http://127.0.0.1:{BACKEND_PORT}")
FRONTEND_PORT = int(os.getenv("PORT", os.getenv("FRONTEND_PORT", "6006")))


def _is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """检测 TCP 端口是否被占用（通过 bind 尝试，比 connect 探测更可靠）。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        return True
    finally:
        sock.close()
    return False


def preflight_check_ports() -> bool:
    """启动前端口预检，避免后续 backend / gradio 因端口占用而失败。"""
    busy = [p for p in (BACKEND_PORT, FRONTEND_PORT) if _is_port_in_use(p)]
    if not busy:
        return True
    error("Main", f"端口被占用: {', '.join(str(p) for p in busy)}，请清理后重试")
    warn("Main", "提示: pkill -9 -f 'python.*main\\.py' && fuser -k 6006/tcp 6008/tcp")
    return False


async def wait_for_server(url: str, max_wait: float = 40.0) -> bool:
    """等待服务端完全启动（使用 socket 检测）"""
    parsed = urlparse(url)
    host = parsed.hostname or 'localhost'
    port = parsed.port or 80

    print(f"  等待 {host}:{port} 就绪 (最多 {max_wait:.0f} 秒)", end="", flush=True)

    start_time = time.time()
    check_interval = 0.2  # 更激进的检测频率
    dot_interval = 1.0
    last_dot_time = start_time

    while time.time() - start_time < max_wait:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                # 端口就绪，给 ASGI lifespan 一点点收尾时间即可
                print(" ✓", flush=True)
                return True
        except Exception:
            pass

        current_time = time.time()
        if current_time - last_dot_time >= dot_interval:
            print(".", end="", flush=True)
            last_dot_time = current_time

        await asyncio.sleep(check_interval)

    print(" ✗", flush=True)
    return False


def start_backend():
    """在独立进程中启动 FastAPI 后端（不再 sleep，由调用方通过端口探测确认就绪）"""
    info("Main", "[1/2] 启动 FastAPI 后端 (内嵌本地 MCP)")
    process = subprocess.Popen(
        [sys.executable, "backend.py"],
        text=True,
        bufsize=1
    )
    # 立即返回，由后续的 wait_for_server 判定真正就绪
    return process


async def main():
    """主函数 - 启动所有服务"""
    init_langfuse()
    banner("Travel Agent 系统启动中")

    backend_process = None

    try:
        # 步骤 0：端口预检，提前发现 6006 / 6008 被占用的情况，避免后端起来后才在 gradio 阶段失败
        if not preflight_check_ports():
            return

        # 步骤 1：启动后端进程（本地 MCP 已内嵌其中）
        backend_process = start_backend()

        # 步骤 2：等待后端就绪（本地 MCP 与 backend 同端口，一次探测即可）
        backend_ready = await wait_for_server(BACKEND_URL, max_wait=40)
        if not backend_ready:
            error("Main", "后端服务未能在 40 秒内启动")
            if backend_process and backend_process.poll() is None:
                backend_process.terminate()
            return

        # 后端进程退出兜底检查
        if backend_process.poll() is not None:
            error("Main", "后端进程已意外退出")
            return

        # 步骤 3：启动 Gradio 前端
        info("Main", "[2/2] 启动 Gradio 前端")

        from gradio_manager import GradioManager

        gradio_manager = GradioManager(backend_url=BACKEND_URL)
        demo = gradio_manager.create_interface()

        ok("Main", "所有服务已启动")
        info("Main", f"前端: http://0.0.0.0:{FRONTEND_PORT}")
        info("Main", f"后端: {BACKEND_URL}")
        info("Main", f"本地 MCP: {BACKEND_URL}/sse (内嵌于后端)")

        demo.launch(
            server_name="0.0.0.0",
            server_port=FRONTEND_PORT,
            share=False,
            show_error=True,
            enable_monitoring=False,
            strict_cors=True,
            max_file_size="1mb",
            footer_links=[],
            theme=gradio_manager.theme,
            css=gradio_manager.css,
            prevent_thread_lock=True,  # 在 asyncio 环境下显式非阻塞，由下方协程统一阻塞
        )

        # 在 asyncio 环境下，gradio 的 launch() 会自动返回非阻塞模式（避免阻塞 jupyter 等场景），
        # 因此这里手动让主协程阻塞，直到收到 Ctrl+C 或后端/前端进程异常退出。
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        import signal as _signal
        for _sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                loop.add_signal_handler(_sig, stop_event.set)
            except NotImplementedError:
                # Windows 等平台不支持 add_signal_handler，则回退到默认 KeyboardInterrupt 机制
                pass

        # 同时监控后端子进程，若其异常退出则主进程也退出
        async def _watch_backend():
            while not stop_event.is_set():
                if backend_process.poll() is not None:
                    error("Main", "检测到后端进程已退出，主程序即将关闭")
                    stop_event.set()
                    return
                await asyncio.sleep(1.0)

        watcher = asyncio.create_task(_watch_backend())
        try:
            await stop_event.wait()
        finally:
            watcher.cancel()

    except KeyboardInterrupt:
        info("Main", "正在关闭所有服务...")
    except Exception as _e:
        import traceback
        error("Main", f"主流程异常: {_e}")
        traceback.print_exc()
    finally:
        flush_langfuse()
        if backend_process and backend_process.poll() is None:
            info("Main", "关闭后端服务...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=3)
            except Exception:
                backend_process.kill()

        ok("Main", "所有服务已关闭")
        os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
