"""Shell 执行工具，支持可选沙箱。"""

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from tools.base import Tool
from logger import info, warn


class ExecTool(Tool):
    """在可选沙箱隔离环境中执行 Shell 命令的工具。"""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        use_sandbox: bool = True,
        sandbox_config: str | None = None,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",          # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        self.use_sandbox = use_sandbox
        self.sandbox_config = sandbox_config

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        if self.use_sandbox:
            return "在隔离沙箱环境中执行 Shell 命令。"
        return "执行 Shell 命令并返回输出。请谨慎使用。"

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 Shell 命令",
                },
                "working_dir": {
                    "type": "string",
                    "description": "命令的可选工作目录",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "超时时间（秒）。对于编译或安装等长时间运行的命令请增大该值"
                        "（默认 60，最大 600）。"
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self, command: str, working_dir: str | None = None,
        timeout: int | None = None, **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        if self.use_sandbox:
            return await self._execute_in_sandbox(command, cwd, effective_timeout)
        else:
            return await self._execute_direct(command, cwd, effective_timeout)

    async def _execute_in_sandbox(
        self, command: str, cwd: str, timeout: int
    ) -> str:
        """使用 srt 在沙箱中执行命令。"""
        # 查找沙箱配置文件
        srt_config = self._find_sandbox_config()
        if not srt_config:
            return "Error: Sandbox enabled but no .srt-settings.json found"

        # 打印沙箱启动信息
        info("Sandbox", f"启动沙箱执行 | cmd={command[:60]}{'…' if len(command) > 60 else ''}")

        # 创建临时脚本文件来执行命令
        # 因为 srt 需要执行一个命令，我们把用户命令写入临时脚本
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
            f.write(f"cd {cwd}\n")
            f.write(f"{command}\n")
            script_path = f.name

        try:
            # 构建 srt 命令: npx srt --settings <config> bash <script>
            cmd_args = [
                "npx", "srt",
                "--settings", srt_config,
                "bash", script_path
            ]

            env = os.environ.copy()
            if self.path_append:
                env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                await self._kill_and_drain(process)
                # 打印沙箱结束信息（超时）
                warn("Sandbox", f"沙箱执行超时（{timeout}秒），已终止")
                return f"Error: Command timed out after {timeout} seconds (in sandbox)"

            return self._format_output(stdout, stderr, process.returncode, sandboxed=True)

        except Exception as e:
            # 打印沙箱错误信息
            warn("Sandbox", f"沙箱执行出错: {str(e)}")
            return f"Error executing command in sandbox: {str(e)}"
        finally:
            # 清理临时脚本
            try:
                os.unlink(script_path)
            except OSError:
                pass

    async def _execute_direct(
        self, command: str, cwd: str, timeout: int
    ) -> str:
        """直接执行命令（不使用沙箱）。"""
        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                await self._kill_and_drain(process)
                return f"Error: Command timed out after {timeout} seconds"

            return self._format_output(stdout, stderr, process.returncode, sandboxed=False)

        except Exception as e:
            return f"Error executing command: {str(e)}"

    async def _kill_and_drain(self, process: asyncio.subprocess.Process) -> None:
        """终止超时子进程并干净地关闭管道传输。"""
        try:
            process.kill()
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(process.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        except (ProcessLookupError, RuntimeError):
            pass
        finally:
            if sys.platform != "win32":
                try:
                    os.waitpid(process.pid, os.WNOHANG)
                except (ProcessLookupError, ChildProcessError):
                    pass

    def _format_output(
        self, stdout: bytes, stderr: bytes, returncode: int, sandboxed: bool
    ) -> str:
        """格式化命令输出。"""
        output_parts = []

        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))

        if stderr:
            stderr_text = stderr.decode("utf-8", errors="replace")
            if stderr_text.strip():
                output_parts.append(f"STDERR:\n{stderr_text}")

        prefix = "[Sandbox] 沙箱执行完成，" if sandboxed else ""
        output_parts.append(f"\n{prefix}Exit code: {returncode}")

        result = "\n".join(output_parts) if output_parts else "(no output)"

        # 采用头尾截断，保留输出开头和结尾
        max_len = self._MAX_OUTPUT
        if len(result) > max_len:
            half = max_len // 2
            result = (
                result[:half]
                + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                + result[-half:]
            )

        return result

    def _find_sandbox_config(self) -> str | None:
        """查找沙箱配置文件。"""
        if self.sandbox_config and os.path.isfile(self.sandbox_config):
            return self.sandbox_config

        # 搜索常见的配置文件位置
        possible_paths = [
            ".srt-settings.json",
            "../.srt-settings.json",
            "../../.srt-settings.json",
            os.path.expanduser("~/.config/travel-agent/.srt-settings.json"),
        ]

        # 从工作目录开始搜索
        search_dir = Path(self.working_dir) if self.working_dir else Path.cwd()
        for path in possible_paths:
            full_path = search_dir / path
            if full_path.is_file():
                return str(full_path.resolve())

        return None

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """尽最大努力防护潜在的危险命令。"""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

        return None
