"""ask_clarification工具，用于向用户收集缺失的信息。"""

import json
from typing import Any

from tools.base import Tool


class AskClarificationTool(Tool):
    """当关键信息缺失或有歧义时，向用户询问澄清的工具。

    LLM 可以调用此工具向用户展示结构化的问题，而不是猜测或做假设。
    在 travel-agent 循环中，调用此工具会中断当前运行，并将格式化的问题作为最终响应返回——
    在用户回复之前，代理不会继续调用其他工具。
    """

    @property
    def name(self) -> str:
        return "ask_clarification"

    @property
    def description(self) -> str:
        return (
            "当用户问题缺失关键信息、有歧义、需要在多种方案中做选择或操作有风险需要确认时，"
            "调用此工具向用户展示结构化的澄清问题。调用后本轮对话将暂停，等待用户回复。"
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "向用户提出的澄清问题",
                },
                "clarification_type": {
                    "type": "string",
                    "enum": [
                        "missing_info",
                        "ambiguous_requirement",
                        "approach_choice",
                        "risk_confirmation",
                        "suggestion",
                    ],
                    "description": (
                        "澄清类型: missing_info=缺少关键参数 | "
                        "ambiguous_requirement=需求有歧义 | "
                        "approach_choice=多种方案需选择 | "
                        "risk_confirmation=操作有风险需确认 | "
                        "suggestion=提供建议供参考"
                    ),
                },
                "context": {
                    "type": "string",
                    "description": "补充背景信息，帮助用户理解为什么需要澄清",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "供用户选择的选项列表，用户可从中选择（最多4项）",
                },
            },
            "required": ["question"],
        }

    async def execute(
        self,
        question: str = "",
        clarification_type: str = "missing_info",
        context: str = "",
        options: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """生成结构化的澄清载荷（JSON格式）。

        生成的 JSON 会被 TravelAgent 的 tool_node 拦截，
        并格式化为人类可读的消息，同时停止代理循环（通过设置 final_response）。
        """
        options = options or []

        payload = {
            "question": question or "为了继续处理，我还需要你补充一些关键信息。",
            "clarification_type": clarification_type,
            "context": context,
            "options": options,
        }
        return json.dumps(payload, ensure_ascii=False)
