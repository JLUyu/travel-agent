"""文件系统工具：读取文件。"""

from pathlib import Path
from typing import Any

from tools.base import Tool


class ReadFileTool(Tool):
    """读取文件内容，支持按行分页。"""

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2000

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None, extra_allowed_dirs: list[Path] | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._extra_allowed_dirs = extra_allowed_dirs

    def _resolve(self, path: str) -> Path:
        """将路径解析到工作区（相对路径），并强制执行目录限制。"""
        p = Path(path).expanduser()
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        resolved = p.resolve()
        if self._allowed_dir:
            all_dirs = [self._allowed_dir] + (self._extra_allowed_dirs or [])
            if not any(self._is_under(resolved, d) for d in all_dirs):
                raise PermissionError(f"Path {path} is outside allowed directory {self._allowed_dir}")
        return resolved

    @staticmethod
    def _is_under(path: Path, directory: Path) -> bool:
        try:
            path.relative_to(directory.resolve())
            return True
        except ValueError:
            return False

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Returns numbered lines. "
            "Use offset and limit to paginate through large files."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要读取的文件路径"},
                "offset": {
                    "type": "integer",
                    "description": "起始行号（从 1 开始，默认为 1）",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "最大读取行数（默认 2000）",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str | None = None, offset: int = 1, limit: int | None = None, **kwargs: Any) -> Any:
        try:
            if not path:
                return "Error reading file: Unknown path"
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"

            raw = fp.read_bytes()
            if not raw:
                return f"(Empty file: {path})"

            try:
                text_content = raw.decode("utf-8")
            except UnicodeDecodeError:
                return f"Error: Cannot read binary file {path}. Only UTF-8 text files are supported."

            all_lines = text_content.splitlines()
            total = len(all_lines)

            if offset < 1:
                offset = 1
            if offset > total:
                return f"Error: offset {offset} is beyond end of file ({total} lines)"

            start = offset - 1
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
            result = "\n".join(numbered)

            if len(result) > self._MAX_CHARS:
                trimmed, chars = [], 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > self._MAX_CHARS:
                        break
                    trimmed.append(line)
                end = start + len(trimmed)
                result = "\n".join(trimmed)

            if end < total:
                result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
            else:
                result += f"\n\n(End of file — {total} lines total)"
            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"
