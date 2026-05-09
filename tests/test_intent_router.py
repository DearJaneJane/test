# tests/test_intent_router.py
"""
intent_router 模块单元测试。

通过 Mock OpenAI 客户端避免真实 API 调用，
验证 12 个典型问题的意图路由结果。
"""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from skill.config import Config
from skill.intent_router import IntentResult, IntentRouter, IntentType, _parse_llm_response


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_config() -> Config:
    """构造测试用 Config。"""
    return Config(
        db_path="./test.db",
        kb_path="./test_knowledge",
        dashscope_api_key="sk-test-key",
    )


def _mock_llm_response(intent: str, db_query_type: str, entities: dict,
                        kb_query_hint: str | None = None,
                        confidence: float = 0.9) -> str:
    """构造 LLM 返回的 JSON 字符串。"""
    return json.dumps({
        "intent": intent,
        "db_query_type": db_query_type,
        "entities": entities,
        "kb_query_hint": kb_query_hint,
        "confidence": confidence,
    }, ensure_ascii=False)


def _create_router_with_mock(llm_json: str) -> IntentRouter:
    """
    创建一个 IntentRouter，其内部 OpenAI 客户端被 Mock，
    返回指定的 JSON 字符串。
    """
    router = IntentRouter(_make_config())

    # Mock OpenAI client 的 chat.completions.create
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = llm_json

    router._client = MagicMock()
    router._client.chat.completions.create.return_value = mock_response

    return router


# ── 意图路由测试 ──────────────────────────────────────────────────────


class TestIntentRouterRoute:
    """测试 IntentRouter.route() 的 12 个典型问题。"""

    def test_01_zhangsan_department(self):
        """张三的部门是什么 → db_only, employee_name=张三"""
        llm_json = _mock_llm_response(
            intent="db_only",
            db_query_type="employee_info",
            entities={"employee_name": "张三", "employee_id": None,
                      "department": None, "year": None, "month": None,
                      "project_status": None},
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("张三的部门是什么")

        assert result.intent == IntentType.DB_ONLY
        assert result.entities["employee_name"] == "张三"
        assert result.db_query_type == "employee_info"

    def test_02_lisi_manager(self):
        """李四的上级是谁 → db_only, employee_name=李四"""
        llm_json = _mock_llm_response(
            intent="db_only",
            db_query_type="employee_info",
            entities={"employee_name": "李四", "employee_id": None,
                      "department": None, "year": None, "month": None,
                      "project_status": None},
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("李四的上级是谁")

        assert result.intent == IntentType.DB_ONLY
        assert result.entities["employee_name"] == "李四"

    def test_03_annual_leave(self):
        """年假怎么计算 → kb_only"""
        llm_json = _mock_llm_response(
            intent="kb_only",
            db_query_type="none",
            entities={"employee_name": None, "employee_id": None,
                      "department": None, "year": None, "month": None,
                      "project_status": None},
            kb_query_hint="年假 计算 天数",
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("年假怎么计算")

        assert result.intent == IntentType.KB_ONLY
        assert result.kb_query_hint  # 非空

    def test_04_late_penalty(self):
        """迟到几次会扣钱 → kb_only"""
        llm_json = _mock_llm_response(
            intent="kb_only",
            db_query_type="none",
            entities={"employee_name": None, "employee_id": None,
                      "department": None, "year": None, "month": None,
                      "project_status": None},
            kb_query_hint="迟到 扣款 次数",
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("迟到几次会扣钱")

        assert result.intent == IntentType.KB_ONLY

    def test_05_zhangsan_projects(self):
        """张三负责哪些项目 → db_only, db_query_type=employee_projects"""
        llm_json = _mock_llm_response(
            intent="db_only",
            db_query_type="employee_projects",
            entities={"employee_name": "张三", "employee_id": None,
                      "department": None, "year": None, "month": None,
                      "project_status": None},
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("张三负责哪些项目")

        assert result.intent == IntentType.DB_ONLY
        assert result.db_query_type == "employee_projects"
        assert result.entities["employee_name"] == "张三"

    def test_06_rd_department_count(self):
        """研发部有多少人 → db_only, db_query_type=dept_members"""
        llm_json = _mock_llm_response(
            intent="db_only",
            db_query_type="dept_members",
            entities={"employee_name": None, "employee_id": None,
                      "department": "研发部", "year": None, "month": None,
                      "project_status": None},
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("研发部有多少人")

        assert result.intent == IntentType.DB_ONLY
        assert result.db_query_type == "dept_members"
        assert result.entities["department"] == "研发部"

    def test_07_wangwu_promotion(self):
        """王五符合P5晋升P6条件吗 → hybrid"""
        llm_json = _mock_llm_response(
            intent="hybrid",
            db_query_type="performance",
            entities={"employee_name": "王五", "employee_id": None,
                      "department": None, "year": None, "month": None,
                      "project_status": None},
            kb_query_hint="P5晋升P6 条件",
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("王五符合P5晋升P6条件吗")

        assert result.intent == IntentType.HYBRID
        assert result.entities["employee_name"] == "王五"

    def test_08_zhangsan_feb_late(self):
        """张三2月迟到几次 → db_only, db_query_type=attendance, month=2"""
        llm_json = _mock_llm_response(
            intent="db_only",
            db_query_type="attendance",
            entities={"employee_name": "张三", "employee_id": None,
                      "department": None, "year": 2026, "month": 2,
                      "project_status": None},
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("张三2月迟到几次")

        assert result.intent == IntentType.DB_ONLY
        assert result.db_query_type == "attendance"
        assert result.entities["month"] == 2

    def test_09_emp_id_query(self):
        """查一下EMP-999 → db_only, employee_id=EMP-999"""
        llm_json = _mock_llm_response(
            intent="db_only",
            db_query_type="employee_info",
            entities={"employee_name": None, "employee_id": "EMP-999",
                      "department": None, "year": None, "month": None,
                      "project_status": None},
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("查一下EMP-999")

        assert result.intent == IntentType.DB_ONLY
        assert result.entities["employee_id"] == "EMP-999"

    def test_10_vague_question(self):
        """最近有什么事 → unclear 或 kb_only"""
        llm_json = _mock_llm_response(
            intent="unclear",
            db_query_type="none",
            entities={"employee_name": None, "employee_id": None,
                      "department": None, "year": None, "month": None,
                      "project_status": None},
            confidence=0.3,
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("最近有什么事")

        assert result.intent in (IntentType.UNCLEAR, IntentType.KB_ONLY)

    def test_11_sql_injection(self):
        """SQL 注入 → injection_detected（不调用 API）"""
        router = _create_router_with_mock("")  # 不应被调用
        result = router.route("SELECT * FROM users WHERE '1'='1'")

        assert result.intent == IntentType.INJECTION_DETECTED
        assert result.confidence == 1.0
        # 确认 API 未被调用
        router._client.chat.completions.create.assert_not_called()

    def test_12_travel_reimbursement(self):
        """差旅费报销标准 → kb_only"""
        llm_json = _mock_llm_response(
            intent="kb_only",
            db_query_type="none",
            entities={"employee_name": None, "employee_id": None,
                      "department": None, "year": None, "month": None,
                      "project_status": None},
            kb_query_hint="差旅费 报销 标准",
        )
        router = _create_router_with_mock(llm_json)
        result = router.route("差旅费报销标准")

        assert result.intent == IntentType.KB_ONLY
        assert result.kb_query_hint  # 非空


# ── 降级规则测试 ──────────────────────────────────────────────────────


class TestFallbackClassify:
    """测试 API 失败时的降级规则。"""

    def _make_failing_router(self) -> IntentRouter:
        """创建一个 API 必定失败的 router。"""
        router = IntentRouter(_make_config())
        router._client = MagicMock()
        router._client.chat.completions.create.side_effect = Exception("API 不可用")
        return router

    def test_fallback_db_with_name(self):
        """包含已知人名 → DB_ONLY。"""
        router = self._make_failing_router()
        result = router.route("张三的部门")
        assert result.intent == IntentType.DB_ONLY
        assert result.entities["employee_name"] == "张三"

    def test_fallback_db_with_keyword(self):
        """包含 DB 关键词 → DB_ONLY。"""
        router = self._make_failing_router()
        result = router.route("研发部有多少人")
        assert result.intent == IntentType.DB_ONLY

    def test_fallback_kb_with_keyword(self):
        """包含 KB 关键词 → KB_ONLY。"""
        router = self._make_failing_router()
        result = router.route("年假怎么计算")
        assert result.intent == IntentType.KB_ONLY

    def test_fallback_hybrid(self):
        """同时包含人名和 KB 关键词 → HYBRID。"""
        router = self._make_failing_router()
        result = router.route("王五符合晋升条件吗")
        assert result.intent == IntentType.HYBRID

    def test_fallback_emp_id(self):
        """包含工号模式 → DB_ONLY。"""
        router = self._make_failing_router()
        result = router.route("查一下EMP-999")
        assert result.intent == IntentType.DB_ONLY
        assert result.entities["employee_id"] == "EMP-999"

    def test_fallback_unclear(self):
        """无任何匹配 → UNCLEAR。"""
        router = self._make_failing_router()
        result = router.route("你好啊")
        assert result.intent == IntentType.UNCLEAR


# ── JSON 解析测试 ─────────────────────────────────────────────────────


class TestParseLlmResponse:
    """测试 _parse_llm_response() 的容错能力。"""

    def test_parse_valid_json(self):
        """合法 JSON → 正确解析。"""
        content = json.dumps({
            "intent": "db_only",
            "db_query_type": "employee_info",
            "entities": {"employee_name": "张三"},
            "confidence": 0.95,
        })
        result = _parse_llm_response(content)
        assert result is not None
        assert result.intent == IntentType.DB_ONLY
        assert result.entities["employee_name"] == "张三"

    def test_parse_markdown_wrapped_json(self):
        """```json ... ``` 包裹的 JSON → 正确解析。"""
        content = '```json\n{"intent": "kb_only", "db_query_type": "none", "entities": {}, "confidence": 0.8}\n```'
        result = _parse_llm_response(content)
        assert result is not None
        assert result.intent == IntentType.KB_ONLY

    def test_parse_invalid_json(self):
        """非法 JSON → 返回 None。"""
        result = _parse_llm_response("这不是JSON")
        assert result is None

    def test_parse_unknown_intent(self):
        """未知 intent 值 → 降级为 UNCLEAR。"""
        content = json.dumps({"intent": "unknown_type", "entities": {}})
        result = _parse_llm_response(content)
        assert result is not None
        assert result.intent == IntentType.UNCLEAR

    def test_parse_string_year_month(self):
        """year/month 为字符串数字 → 正确转为 int。"""
        content = json.dumps({
            "intent": "db_only",
            "db_query_type": "attendance",
            "entities": {"year": "2026", "month": "2"},
            "confidence": 0.9,
        })
        result = _parse_llm_response(content)
        assert result is not None
        assert result.entities["year"] == 2026
        assert result.entities["month"] == 2
