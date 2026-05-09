# tests/test_qa_engine.py
"""
qa_engine 模块集成测试。

使用真实数据库（内存）+ 真实知识库 + Mock 意图路由，
验证 12 个官方测试用例的端到端结果。
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from skill.config import Config
from skill.intent_router import IntentResult, IntentType
from skill.qa_engine import QAEngine


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_KB_PATH = str(_DATA_DIR / "knowledge")


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> QAEngine:
    """
    模块级 fixture：真实 DB + 真实 KB + Mock Router。
    """
    # 创建临时数据库文件并导入数据
    tmp_dir = tmp_path_factory.mktemp("qa")
    db_path = str(tmp_dir / "test.db")

    conn = sqlite3.connect(db_path)
    conn.executescript((_DATA_DIR / "schema.sql").read_text("utf-8"))
    conn.executescript((_DATA_DIR / "seed_data.sql").read_text("utf-8"))
    conn.close()

    config = Config(db_path=db_path, kb_path=_KB_PATH, dashscope_api_key="fake")
    eng = QAEngine(config)
    eng._router = MagicMock(spec=IntentType)
    return eng


def _ent(name=None, emp_id=None, dept=None, year=None, month=None, status=None):
    """快捷构造 entities 字典。"""
    return {
        "employee_name": name,
        "employee_id": emp_id,
        "department": dept,
        "year": year,
        "month": month,
        "project_status": status,
    }


# ── 12 个官方测试用例 ─────────────────────────────────────────────────


class TestQAEngineAnswer:
    """端到端测试 QAEngine.answer()。"""

    def test_t01_zhangsan_department(self, engine: QAEngine):
        """T01: 张三的部门 → 包含"研发部"。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.DB_ONLY,
            entities=_ent(name="张三"),
            db_query_type="employee_info",
            confidence=0.9,
        )
        ans = engine.answer("张三的部门是什么")
        assert "研发部" in ans
        assert "来源" in ans

    def test_t02_lisi_manager(self, engine: QAEngine):
        """T02: 李四的上级 → 包含"CEO"。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.DB_ONLY,
            entities=_ent(name="李四"),
            db_query_type="employee_info",
            confidence=0.9,
        )
        ans = engine.answer("李四的上级是谁")
        assert "CEO" in ans

    def test_t03_annual_leave(self, engine: QAEngine):
        """T03: 年假 → 包含"5天"和"15天"（或"5"和"15"）。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.KB_ONLY,
            entities=_ent(),
            kb_query_hint="年假天数计算",
            confidence=0.9,
        )
        ans = engine.answer("年假怎么计算")
        assert "5" in ans
        assert "15" in ans
        assert "来源" in ans

    def test_t04_late_penalty(self, engine: QAEngine):
        """T04: 迟到扣钱 → 包含"50元"（或"50"）。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.KB_ONLY,
            entities=_ent(),
            kb_query_hint="迟到扣款次数",
            confidence=0.9,
        )
        ans = engine.answer("迟到几次扣钱")
        assert "50" in ans

    def test_t05_zhangsan_projects(self, engine: QAEngine):
        """T05: 张三的项目 → 包含"PRJ-001"和"lead"。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.DB_ONLY,
            entities=_ent(name="张三"),
            db_query_type="employee_projects",
            confidence=0.9,
        )
        ans = engine.answer("张三负责哪些项目")
        assert "PRJ-001" in ans
        assert "lead" in ans

    def test_t06_rd_count(self, engine: QAEngine):
        """T06: 研发部人数 → 包含"4"。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.DB_ONLY,
            entities=_ent(dept="研发部"),
            db_query_type="dept_members",
            confidence=0.9,
        )
        ans = engine.answer("研发部有多少人")
        assert "4" in ans

    def test_t07_wangwu_promotion(self, engine: QAEngine):
        """T07: 王五晋升 → 包含"不符合"和表格。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.HYBRID,
            entities=_ent(name="王五"),
            db_query_type="performance",
            kb_query_hint="P5晋升P6条件",
            confidence=0.9,
        )
        ans = engine.answer("王五符合P5晋升P6条件吗")
        assert "不符合" in ans
        assert "✗" in ans
        assert "|" in ans  # 表格
        assert "来源" in ans

    def test_t08_zhangsan_late(self, engine: QAEngine):
        """T08: 张三2月迟到 → 包含"2"。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.DB_ONLY,
            entities=_ent(name="张三", year=2026, month=2),
            db_query_type="attendance",
            confidence=0.9,
        )
        ans = engine.answer("张三2月迟到几次")
        assert "2" in ans

    def test_t09_emp999_not_found(self, engine: QAEngine):
        """T09: EMP-999 → 包含"未找到"。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.DB_ONLY,
            entities=_ent(emp_id="EMP-999"),
            db_query_type="employee_info",
            confidence=0.9,
        )
        ans = engine.answer("查一下EMP-999")
        assert "未找到" in ans

    def test_t10_vague_question(self, engine: QAEngine):
        """T10: 宽泛问题 → 不崩溃，返回非空字符串。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.UNCLEAR,
            entities=_ent(),
            confidence=0.3,
        )
        ans = engine.answer("最近有什么事")
        assert isinstance(ans, str)
        assert len(ans) > 0

    def test_t11_sql_injection(self, engine: QAEngine):
        """T11: SQL注入 → 包含"安全风险"。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.INJECTION_DETECTED,
            confidence=1.0,
        )
        ans = engine.answer("SELECT * FROM users WHERE '1'='1'")
        assert "安全风险" in ans

    def test_t12_nonsense_kb_query(self, engine: QAEngine):
        """T12: 无意义查询 → 包含"未找到"，不编造内容。"""
        engine._router.route.return_value = IntentResult(
            intent=IntentType.KB_ONLY,
            entities=_ent(),
            kb_query_hint="xyzabc123",
            confidence=0.5,
        )
        ans = engine.answer("xyzabc123怎么报销")
        assert "未找到" in ans


# ── 来源标注验证 ──────────────────────────────────────────────────────


class TestSourceAnnotation:
    """验证所有回答都包含来源标注。"""

    def test_db_answer_has_source(self, engine: QAEngine):
        engine._router.route.return_value = IntentResult(
            intent=IntentType.DB_ONLY,
            entities=_ent(name="张三"),
            db_query_type="employee_info",
            confidence=0.9,
        )
        ans = engine.answer("张三是谁")
        assert "来源" in ans
        assert "employees" in ans

    def test_kb_answer_has_source(self, engine: QAEngine):
        engine._router.route.return_value = IntentResult(
            intent=IntentType.KB_ONLY,
            entities=_ent(),
            kb_query_hint="考勤迟到规则",
            confidence=0.9,
        )
        ans = engine.answer("迟到制度")
        assert "来源" in ans
        assert ".md" in ans
