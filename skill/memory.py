"""Conversation memory helpers for interactive multi-turn QA."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from skill.intent_router import IntentResult, IntentType


_KNOWN_EMPLOYEES = {
    "张三": "EMP-001",
    "李四": "EMP-002",
    "王五": "EMP-003",
    "赵六": "EMP-004",
    "钱七": "EMP-005",
    "孙八": "EMP-006",
    "周九": "EMP-007",
    "吴十": "EMP-008",
    "CEO": "EMP-000",
}

_FOLLOWUP_MARKERS = ("呢", "那", "他", "她", "这个人", "同样", "也")


@dataclass
class ConversationTurn:
    question: str
    intent: IntentResult


class ConversationMemory:
    """Keeps the most recent resolved turns in process memory."""

    def __init__(self, max_turns: int = 5) -> None:
        self._turns: deque[ConversationTurn] = deque(maxlen=max(1, max_turns))

    @property
    def max_turns(self) -> int:
        return self._turns.maxlen or 0

    def remember(self, question: str, intent: IntentResult) -> None:
        if intent.intent in {IntentType.INJECTION_DETECTED, IntentType.UNCLEAR}:
            return
        self._turns.append(ConversationTurn(question=question, intent=deepcopy(intent)))

    def resolve_followup(self, question: str, intent: IntentResult) -> IntentResult:
        """Inherit the previous query type for short follow-up questions."""
        if not self._turns:
            return intent

        name, employee_id = self._extract_employee(question)
        looks_like_followup = any(marker in question for marker in _FOLLOWUP_MARKERS)
        if not looks_like_followup and not (name and intent.intent == IntentType.UNCLEAR):
            return intent

        previous = self._turns[-1].intent
        if previous.intent not in {IntentType.DB_ONLY, IntentType.HYBRID}:
            return intent

        resolved = deepcopy(previous)
        resolved.confidence = min(previous.confidence, 0.65)
        if name:
            resolved.entities["employee_name"] = name
            resolved.entities["employee_id"] = employee_id
        self._merge_current_entities(resolved.entities, intent.entities)
        return resolved

    @staticmethod
    def _extract_employee(question: str) -> tuple[str | None, str | None]:
        for name, employee_id in _KNOWN_EMPLOYEES.items():
            if name in question:
                return name, employee_id
        return None, None

    @staticmethod
    def _merge_current_entities(target: dict[str, Any], current: dict[str, Any]) -> None:
        for key in ("department", "year", "month", "project_status"):
            value = current.get(key)
            if value is not None:
                target[key] = value
