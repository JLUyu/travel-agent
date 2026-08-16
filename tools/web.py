"""受限的只读网页搜索与抓取工具。"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from tools.base import Tool


_USER_AGENT = (
    "Mozilla/5.0 (compatible; TravelAgent/1.0; "
    "+https://github.com/JLUyu/travel-agent)"
)
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
_ALLOWED_PORTS = {None, 80, 443}
_TEXT_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
)


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def _unwrap_duckduckgo_url(url: str) -> str:
    absolute = urljoin("https://duckduckgo.com", html.unescape(url))
    parsed = urlparse(absolute)
    target = parse_qs(parsed.query).get("uddg", [""])[0]
    return unquote(target) if target else absolute


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._capture: str | None = None
        self._buffer: list[str] = []
        self._current: dict[str, str] | None = None

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        value = dict(attrs).get("class") or ""
        return set(value.split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = self._classes(attrs)
        if tag == "a" and "result__a" in classes:
            href = dict(attrs).get("href") or ""
            self._current = {"title": "", "url": _unwrap_duckduckgo_url(href), "snippet": ""}
            self._capture = "title"
            self._buffer = []
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture == "title" and tag == "a" and self._current is not None:
            self._current["title"] = _normalize_space("".join(self._buffer))
            self.results.append(self._current)
            self._capture = None
            self._buffer = []
        elif self._capture == "snippet" and tag in {"a", "div"} and self._current is not None:
            self._current["snippet"] = _normalize_space("".join(self._buffer))
            self._capture = None
            self._buffer = []


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True
        elif not self._ignored_depth and tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "article", "section"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.text_parts.append(data)

    @property
    def title(self) -> str:
        return _normalize_space("".join(self.title_parts))

    @property
    def text(self) -> str:
        value = html.unescape(" ".join(self.text_parts))
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        return re.sub(r"\n{3,}", "\n\n", value).strip()


async def _validate_public_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("仅支持 http/https URL")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL 主机无效或包含凭据")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL 端口无效") from exc
    if port not in _ALLOWED_PORTS:
        raise ValueError("仅允许访问标准 Web 端口 80/443")

    try:
        literal = ipaddress.ip_address(parsed.hostname)
        addresses = [literal]
    except ValueError:
        infos = await asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
        addresses = list({ipaddress.ip_address(info[4][0]) for info in infos})
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("拒绝访问本机、私网或保留地址")
    return parsed.geturl()


class WebSearchTool(Tool):
    def __init__(self, endpoint: str | None = None, timeout: float = 15) -> None:
        self.endpoint = endpoint or "https://html.duckduckgo.com/html/"
        self.timeout = float(timeout)

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "搜索公开互联网，返回网页标题、链接和摘要。涉及最新信息时优先调用。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词", "minLength": 1, "maxLength": 300},
                "max_results": {"type": "integer", "description": "结果数量，默认 5", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str, max_results: int = 5) -> str:
        max_results = max(1, min(int(max_results), 10))
        await _validate_public_url(self.endpoint)
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=False,
        ) as client:
            response = await client.get(self.endpoint, params={"q": query})
            response.raise_for_status()
        parser = _DuckDuckGoParser()
        parser.feed(response.text)
        results = [item for item in parser.results if item["title"] and item["url"]][:max_results]
        if not results:
            return json.dumps({"query": query, "results": [], "message": "未找到结果"}, ensure_ascii=False)
        return json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2)


class WebFetchTool(Tool):
    def __init__(self, timeout: float = 15) -> None:
        self.timeout = float(timeout)

    @property
    def name(self) -> str:
        return "fetch_url"

    @property
    def description(self) -> str:
        return "读取公开网页的正文。先用 web_search 获取 URL，再用本工具核实详情。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要读取的公开 http/https URL", "minLength": 8, "maxLength": 2048},
                "max_chars": {"type": "integer", "description": "最多返回字符数，默认 10000", "minimum": 1000, "maximum": 20000},
            },
            "required": ["url"],
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, url: str, max_chars: int = 10000) -> str:
        current_url = await _validate_public_url(url)
        max_chars = max(1000, min(int(max_chars), 20000))
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain,application/json,application/xml"},
            follow_redirects=False,
        ) as client:
            for _ in range(4):
                async with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("重定向缺少 Location")
                        current_url = await _validate_public_url(urljoin(current_url, location))
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not content_type.startswith(_TEXT_CONTENT_TYPES):
                        raise ValueError(f"不支持的内容类型: {content_type.split(';')[0]}")
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > _MAX_DOWNLOAD_BYTES:
                            raise ValueError("网页内容超过 2 MB 限制")
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    encoding = response.encoding or "utf-8"
                    body = raw.decode(encoding, errors="replace")
                    if "html" in content_type or "<html" in body[:500].lower():
                        parser = _ReadableHTMLParser()
                        parser.feed(body)
                        title, text = parser.title, parser.text
                    else:
                        title, text = "", _normalize_space(body)
                    return json.dumps(
                        {"url": str(response.url), "title": title, "content": text[:max_chars]},
                        ensure_ascii=False,
                        indent=2,
                    )
            raise ValueError("重定向次数过多")
