from skill.intent_router import IntentResult, IntentType
from skill.memory import ConversationMemory


def test_memory_resolves_employee_followup() -> None:
    memory = ConversationMemory(max_turns=3)
    memory.remember(
        "张三的上级是谁",
        IntentResult(
            intent=IntentType.DB_ONLY,
            db_query_type="employee_info",
            entities={"employee_name": "张三", "employee_id": "EMP-001"},
            confidence=0.9,
        ),
    )

    resolved = memory.resolve_followup(
        "那李四呢",
        IntentResult(intent=IntentType.UNCLEAR, db_query_type="none", confidence=0.3),
    )

    assert resolved.intent == IntentType.DB_ONLY
    assert resolved.db_query_type == "employee_info"
    assert resolved.entities["employee_name"] == "李四"
    assert resolved.entities["employee_id"] == "EMP-002"
