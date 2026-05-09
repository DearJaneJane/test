# tests/test_db_query.py
"""
db_query 模块单元测试。

使用临时数据库文件，导入真实的 schema.sql + seed_data.sql 作为测试数据。
"""

import sqlite3
from pathlib import Path

import pytest

from skill.db_query import (
    check_sql_injection,
    get_attendance_late_count,
    get_connection,
    get_department_members,
    get_employee_by_id,
    get_employee_by_name,
    get_employee_projects,
    get_performance_reviews,
    get_projects_by_status,
)


# ── Fixtures ──────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def db_conn() -> sqlite3.Connection:
    """
    模块级 fixture：在内存中创建数据库并导入真实数据。
    所有测试共享同一个连接（只读操作，无状态污染风险）。
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    schema_sql = (_DATA_DIR / "schema.sql").read_text(encoding="utf-8")
    seed_sql = (_DATA_DIR / "seed_data.sql").read_text(encoding="utf-8")

    conn.executescript(schema_sql)
    conn.executescript(seed_sql)

    yield conn
    conn.close()


# ── 员工查询测试 ──────────────────────────────────────────────────────


class TestGetEmployeeByName:
    """测试 get_employee_by_name()。"""

    def test_zhangsan_department(self, db_conn: sqlite3.Connection):
        """查张三 → 研发部。"""
        emp = get_employee_by_name(db_conn, "张三")
        assert emp is not None
        assert emp["department"] == "研发部"
        assert emp["employee_id"] == "EMP-001"
        assert emp["level"] == "P6"

    def test_zhangsan_manager_is_ceo(self, db_conn: sqlite3.Connection):
        """张三的上级 → CEO。"""
        emp = get_employee_by_name(db_conn, "张三")
        assert emp is not None
        assert emp["manager_name"] == "CEO"

    def test_ceo_has_no_manager(self, db_conn: sqlite3.Connection):
        """CEO 没有上级 → manager_name 为 None。"""
        emp = get_employee_by_name(db_conn, "CEO")
        assert emp is not None
        assert emp["manager_name"] is None

    def test_nonexistent_employee(self, db_conn: sqlite3.Connection):
        """查不存在的员工 → 返回 None。"""
        emp = get_employee_by_name(db_conn, "不存在的人")
        assert emp is None


class TestGetEmployeeById:
    """测试 get_employee_by_id()。"""

    def test_emp002_is_lisi(self, db_conn: sqlite3.Connection):
        """EMP-002 → 李四。"""
        emp = get_employee_by_id(db_conn, "EMP-002")
        assert emp is not None
        assert emp["name"] == "李四"
        assert emp["department"] == "研发部"

    def test_emp002_manager_is_ceo(self, db_conn: sqlite3.Connection):
        """李四(EMP-002)的上级 → CEO。"""
        emp = get_employee_by_id(db_conn, "EMP-002")
        assert emp is not None
        assert emp["manager_name"] == "CEO"

    def test_nonexistent_id(self, db_conn: sqlite3.Connection):
        """查不存在的工号 → 返回 None。"""
        emp = get_employee_by_id(db_conn, "EMP-999")
        assert emp is None


# ── 部门查询测试 ──────────────────────────────────────────────────────


class TestGetDepartmentMembers:
    """测试 get_department_members()。"""

    def test_rd_department_active_count(self, db_conn: sqlite3.Connection):
        """研发部在职人数 → 4（EMP-009 已离职不计入）。"""
        members = get_department_members(db_conn, "研发部")
        assert len(members) == 4

    def test_rd_department_excludes_resigned(self, db_conn: sqlite3.Connection):
        """研发部结果不含离职员工 EMP-009。"""
        members = get_department_members(db_conn, "研发部")
        ids = [m["employee_id"] for m in members]
        assert "EMP-009" not in ids

    def test_rd_department_includes_correct_members(self, db_conn: sqlite3.Connection):
        """研发部包含 EMP-001, EMP-002, EMP-005, EMP-007。"""
        members = get_department_members(db_conn, "研发部")
        ids = sorted(m["employee_id"] for m in members)
        assert ids == ["EMP-001", "EMP-002", "EMP-005", "EMP-007"]

    def test_empty_department(self, db_conn: sqlite3.Connection):
        """不存在的部门 → 返回空列表。"""
        members = get_department_members(db_conn, "不存在的部门")
        assert members == []


# ── 考勤查询测试 ──────────────────────────────────────────────────────


class TestGetAttendanceLateCount:
    """测试 get_attendance_late_count()。"""

    def test_zhangsan_feb_2026_late_count(self, db_conn: sqlite3.Connection):
        """张三 2026 年 2 月迟到次数 → 2。"""
        count = get_attendance_late_count(db_conn, "EMP-001", 2026, 2)
        assert count == 2

    def test_wangwu_feb_2026_late_count(self, db_conn: sqlite3.Connection):
        """王五 2026 年 2 月迟到次数 → 5。"""
        count = get_attendance_late_count(db_conn, "EMP-003", 2026, 2)
        assert count == 5

    def test_no_attendance_data(self, db_conn: sqlite3.Connection):
        """无考勤数据的月份 → 返回 0。"""
        count = get_attendance_late_count(db_conn, "EMP-001", 2026, 1)
        assert count == 0


# ── 绩效查询测试 ──────────────────────────────────────────────────────


class TestGetPerformanceReviews:
    """测试 get_performance_reviews()。"""

    def test_wangwu_2025_reviews(self, db_conn: sqlite3.Connection):
        """王五 2025 年绩效 → 2 条记录（Q3, Q4），平均 80.0。"""
        reviews = get_performance_reviews(db_conn, "EMP-003", 2025)
        assert len(reviews) == 2
        avg_kpi = sum(r["kpi_score"] for r in reviews) / len(reviews)
        assert round(avg_kpi, 1) == 80.0

    def test_zhangsan_2025_reviews(self, db_conn: sqlite3.Connection):
        """张三 2025 年绩效 → 4 条记录，按季度排序。"""
        reviews = get_performance_reviews(db_conn, "EMP-001", 2025)
        assert len(reviews) == 4
        quarters = [r["quarter"] for r in reviews]
        assert quarters == [1, 2, 3, 4]

    def test_no_reviews(self, db_conn: sqlite3.Connection):
        """无绩效数据 → 返回空列表。"""
        reviews = get_performance_reviews(db_conn, "EMP-001", 2020)
        assert reviews == []


# ── 项目查询测试 ──────────────────────────────────────────────────────


class TestGetEmployeeProjects:
    """测试 get_employee_projects()。"""

    def test_zhangsan_project_count(self, db_conn: sqlite3.Connection):
        """张三参与的项目数 → 4。"""
        projects = get_employee_projects(db_conn, "EMP-001")
        assert len(projects) == 4

    def test_zhangsan_lead_projects_first(self, db_conn: sqlite3.Connection):
        """张三的项目按 role 排序，lead 项目在前。"""
        projects = get_employee_projects(db_conn, "EMP-001")
        roles = [p["role"] for p in projects]
        # lead 应该排在 core 和 contributor 之前
        lead_indices = [i for i, r in enumerate(roles) if r == "lead"]
        non_lead_indices = [i for i, r in enumerate(roles) if r != "lead"]
        if lead_indices and non_lead_indices:
            assert max(lead_indices) < min(non_lead_indices)

    def test_no_projects(self, db_conn: sqlite3.Connection):
        """无项目的员工 → 返回空列表。"""
        projects = get_employee_projects(db_conn, "EMP-009")
        assert projects == []


class TestGetProjectsByStatus:
    """测试 get_projects_by_status()。"""

    def test_active_projects(self, db_conn: sqlite3.Connection):
        """active 项目 → 2 个（PRJ-001, PRJ-003）。"""
        projects = get_projects_by_status(db_conn, "active")
        assert len(projects) == 2
        ids = [p["project_id"] for p in projects]
        assert "PRJ-001" in ids
        assert "PRJ-003" in ids

    def test_completed_projects(self, db_conn: sqlite3.Connection):
        """completed 项目 → 1 个。"""
        projects = get_projects_by_status(db_conn, "completed")
        assert len(projects) == 1


# ── SQL 注入检测测试 ──────────────────────────────────────────────────


class TestCheckSqlInjection:
    """测试 check_sql_injection()。"""

    @pytest.mark.parametrize(
        "text",
        [
            "SELECT * FROM users",
            "select * from users",
            "1; DROP TABLE employees",
            "' OR 1=1 --",
            "admin' UNION SELECT * FROM passwords",
            "INSERT INTO logs VALUES('hack')",
            "UPDATE employees SET salary=0",
            "DELETE FROM attendance",
            "test' OR '1'='1",
        ],
    )
    def test_detects_injection(self, text: str):
        """已知注入模式 → 返回 True。"""
        assert check_sql_injection(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "张三的部门",
            "研发部有多少人",
            "王五2月迟到了几次",
            "请假制度是什么",
            "2025年绩效怎么样",
        ],
    )
    def test_safe_input(self, text: str):
        """正常中文问题 → 返回 False。"""
        assert check_sql_injection(text) is False
