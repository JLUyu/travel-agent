"""技能加载器，用于加载 Agent 的能力技能。"""

import json
import os
import re
import shutil
from pathlib import Path

# skill的默认目录（相对于项目根目录）
DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"


class SkillsLoader:
    """
    Agent 技能加载器。

    技能是 Markdown 文件（SKILL.md），教会 Agent 如何使用
    特定工具或执行特定任务。
    """

    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir = skills_dir or DEFAULT_SKILLS_DIR

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        列出所有可用技能。

        Args:
            filter_unavailable: 若为 True，过滤掉依赖未满足的技能。

        Returns:
            技能信息字典列表，包含 'name' 和 'path'。
        """
        skills = []

        # 扫描技能目录
        if self.skills_dir and self.skills_dir.exists():
            for skill_dir in self.skills_dir.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "path": str(skill_file)})

        # 按依赖要求过滤
        if filter_unavailable:
            return [s for s in skills if self._check_requirements(self._get_skill_meta(s["name"]))]
        return skills

    def load_skill(self, name: str) -> str | None:
        """
        按名称加载技能。

        Args:
            name: 技能名称（目录名）。

        Returns:
            技能内容，未找到则返回 None。
        """
        skill_file = self.skills_dir / name / "SKILL.md"
        if skill_file.exists():
            return skill_file.read_text(encoding="utf-8")
        return None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        加载指定技能，用于注入 Agent 上下文。

        Args:
            skill_names: 需要加载的技能名称列表。

        Returns:
            格式化后的技能内容。
        """
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def build_skills_summary(self) -> str:
        """
        构建所有技能的摘要（名称、描述、路径、可用性）。

        用于渐进式加载——Agent 可在需要时通过 read_file 读取完整技能内容。

        Returns:
            XML 格式的技能摘要。
        """
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            name = escape_xml(s["name"])
            path = s["path"]
            desc = escape_xml(self._get_skill_description(s["name"]))
            skill_meta = self._get_skill_meta(s["name"])
            available = self._check_requirements(skill_meta)

            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")

            # 为不可用技能展示缺失的依赖要求
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")

            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """获取缺失的依赖要求描述。"""
        missing = []
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                missing.append(f"CLI: {b}")
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)

    def _get_skill_description(self, name: str) -> str:
        """从 frontmatter 中获取技能描述。"""
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # 返回技能名称

    def _strip_frontmatter(self, content: str) -> str:
        """移除 Markdown 内容中的 YAML frontmatter。"""
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end():].strip()
        return content

    def _parse_agent_metadata(self, raw: str) -> dict:
        """从 frontmatter 解析技能元数据 JSON（支持 agent 键）。"""
        try:
            data = json.loads(raw)
            return data.get("agent", {}) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        """检查技能依赖是否满足（可执行文件、环境变量）。"""
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True

    def _get_skill_meta(self, name: str) -> dict:
        """获取技能的 agent 元数据（从 frontmatter 缓存）。"""
        meta = self.get_skill_metadata(name) or {}
        return self._parse_agent_metadata(meta.get("metadata", ""))

    def get_always_skills(self) -> list[str]:
        """获取标记为 always=true 且依赖已满足的技能。"""
        result = []
        for s in self.list_skills(filter_unavailable=True):
            meta = self.get_skill_metadata(s["name"]) or {}
            # 同时检查顶层 always 和 metadata 中的 always
            skill_meta = self._parse_agent_metadata(meta.get("metadata", ""))
            if skill_meta.get("always") or meta.get("always"):
                result.append(s["name"])
        return result

    def get_skill_metadata(self, name: str) -> dict | None:
        """
        从技能的 frontmatter 获取元数据。

        Args:
            name: 技能名称。

        Returns:
            元数据字典或 None。
        """
        content = self.load_skill(name)
        if not content:
            return None

        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                # 解析 YAML
                metadata = {}
                for line in match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"\'')
                return metadata

        return None
