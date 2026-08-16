"""
统一日志打印工具

提供整齐对齐的彩色日志输出，所有模块的 print 输出格式应通过本模块进行。

输出格式:
    [LEVEL] [Module  ] message

约定:
    - LEVEL 固定 4 字符: INFO / OK   / WARN / ERR  / DEBG
    - Module 默认填充至 10 字符
    - message 应当简短，避免超过 120 字符；过长内容请截断
"""
from __future__ import annotations

import sys
from typing import Any

MODULE_WIDTH = 10
MAX_MSG_LEN = 160
LINE_WIDTH = 60

_USE_COLOR = sys.stdout.isatty()

_COLORS = {
    "INFO": "\033[36m",
    "OK":   "\033[32m",
    "WARN": "\033[33m",
    "ERR":  "\033[31m",
    "DEBG": "\033[90m",
}
_RESET = "\033[0m"
_DIM = "\033[2m"


def _color(level: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{_COLORS.get(level, '')}{text}{_RESET}"


def _truncate(msg: str, limit: int = MAX_MSG_LEN) -> str:
    msg = str(msg).replace("\n", " ").rstrip()
    if len(msg) > limit:
        return msg[: limit - 1] + "…"
    return msg


def _emit(level: str, module: str, message: Any, *, file=sys.stdout) -> None:
    level_tag = f"{level:<4}"
    module_tag = f"{module:<{MODULE_WIDTH}}"[:MODULE_WIDTH]
    msg = _truncate(message)
    prefix = _color(level, f"[{level_tag}] [{module_tag}]")
    print(f"{prefix} {msg}", file=file, flush=True)


def info(module: str, message: Any) -> None:
    _emit("INFO", module, message)


def ok(module: str, message: Any) -> None:
    _emit("OK",   module, message)


def warn(module: str, message: Any) -> None:
    _emit("WARN", module, message, file=sys.stderr)


def error(module: str, message: Any) -> None:
    _emit("ERR",  module, message, file=sys.stderr)


def debug(module: str, message: Any) -> None:
    _emit("DEBG", module, message)


def banner(title: str, *, char: str = "=", width: int = LINE_WIDTH) -> None:
    """打印带标题的横幅，例如:
    ============================================================
                       Travel Agent 启动中
    ============================================================
    """
    line = char * width
    title = f" {title.strip()} "
    print(line, flush=True)
    print(title.center(width), flush=True)
    print(line, flush=True)


def section(title: str, *, char: str = "-", width: int = LINE_WIDTH) -> None:
    """打印分节标题:
    ------------------------ 标题 ------------------------
    """
    title = f" {title.strip()} "
    pad = (width - len(title)) // 2
    if pad < 3:
        pad = 3
    print(char * pad + title + char * (width - pad - len(title)), flush=True)


def rule(*, char: str = "-", width: int = LINE_WIDTH) -> None:
    """打印一条分隔线"""
    print(char * width, flush=True)


def kv(module: str, **fields: Any) -> None:
    """以 key=value 形式打印一组字段，便于阅读"""
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    info(module, parts)
