# skill/db_query.py
"""
数据库查询模块。

封装所有 SQLite 查询操作，所有 SQL 均使用参数化查询（? 占位符），
禁止任何形式的字符串拼接。
"""

import re
import sqlite3
from typing import Any


# ── SQL 注入检测 ──────────────────────────────────────────────────────

# 高风险关键词/模式（大小写不敏感匹配）
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bSELECT\b.+\bFROM\b", re.IGNORECASE),
    re.compile(r"\bDROP\s+(TABLE|DATABASE|INDEX)\b", re.IGNORECASE),
    re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+\w+\s+SET\b", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"\bUNION\s+SELECT\b", re.IGNORECASE),
    re.compile(r"--"),
    re.compile(r"['\"]1['\"]=['\"]1['\"]", re.IGNORECASE),  # '1'='1' 或 "1"="1"
    re.compile(r"\bOR\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"]?", re.IGNORECASE),
]


def check_sql_injection(text: str) -> bool:
    """
    检测输入文本是否包含 SQL 注入特征。

    检测规则（大小写不敏感）：
    - SQL 关键词: SELECT, DROP, INSERT, UPDATE, DELETE, UNION
    - 注释符: --
    - 恒真条件: '1'='1', OR 1=1

    Args:
        text: 待检测的用户输入文本。

    Returns:
        True 表示检测到注入风险，False 表示安全。
    """
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


# ── 连接工具 ──────────────────────────────────────────────────────────


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    获取 SQLite 连接，结果以 dict 形式返回（Row 工厂）。

    Args:
        db_path: 数据库文件路径。

    Returns:
        配置好 Row 工厂的 SQLite 连接。
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """将 sqlite3.Row 转为普通 dict，None 输入返回 None。"""
    if row is None:
        return None
    return dict(row)


def _rows_to_list(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """将 sqlite3.Row 列表转为 dict 列表。"""
    return [dict(r) for r in rows]


# ── 员工查询 ──────────────────────────────────────────────────────────


def get_employee_by_name(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    """
    按姓名查询员工信息（含上级姓名）。

    通过 LEFT JOIN 关联 employees 表自身，获取 manager 的 name。

    Args:
        conn: 数据库连接。
        name: 员工姓名。

    Returns:
        包含 employee_id, name, department, level, hire_date, email,
        status, manager_name 的字典；未找到返回 None。
    """
    sql = """
        SELECT
            e.employee_id,
            e.name,
            e.department,
            e.level,
            e.hire_date,
            e.email,
            e.status,
            m.name AS manager_name
        FROM employees e
        LEFT JOIN employees m ON e.manager_id = m.employee_id
        WHERE e.name = ?
    """
    row = conn.execute(sql, (name,)).fetchone()
    return _row_to_dict(row)


def get_employee_by_id(conn: sqlite3.Connection, employee_id: str) -> dict[str, Any] | None:
    """
    按工号查询员工信息（含上级姓名）。

    通过 LEFT JOIN 关联 employees 表自身，获取 manager 的 name。

    Args:
        conn: 数据库连接。
        employee_id: 员工工号，如 'EMP-001'。

    Returns:
        包含 employee_id, name, department, level, hire_date, email,
        status, manager_name 的字典；未找到返回 None。
    """
    sql = """
        SELECT
            e.employee_id,
            e.name,
            e.department,
            e.level,
            e.hire_date,
            e.email,
            e.status,
            m.name AS manager_name
        FROM employees e
        LEFT JOIN employees m ON e.manager_id = m.employee_id
        WHERE e.employee_id = ?
    """
    row = conn.execute(sql, (employee_id,)).fetchone()
    return _row_to_dict(row)


def get_department_members(conn: sqlite3.Connection, department: str) -> list[dict[str, Any]]:
    """
    查询某部门所有在职员工（status='active'）。

    Args:
        conn: 数据库连接。
        department: 部门名称，如 '研发部'。

    Returns:
        在职员工字典列表，按 employee_id 排序。
        每项包含 employee_id, name, level, hire_date。
        离职员工（status!='active'）已被过滤。
    """
    sql = """
        SELECT employee_id, name, level, hire_date
        FROM employees
        WHERE department = ? AND status = ?
        ORDER BY employee_id
    """
    rows = conn.execute(sql, (department, "active")).fetchall()
    return _rows_to_list(rows)


# ── 项目查询 ──────────────────────────────────────────────────────────


def get_employee_projects(conn: sqlite3.Connection, employee_id: str) -> list[dict[str, Any]]:
    """
    查询员工参与的所有项目。

    通过 JOIN project_members 和 projects 表关联查询。
    按 role 排序：lead 在前（利用字典序 contributor > core > lead 的反序）。

    Args:
        conn: 数据库连接。
        employee_id: 员工工号。

    Returns:
        项目字典列表，每项包含 project_id, project_name, role, project_status。
        按 role 排序（lead → core → contributor）。
    """
    sql = """
        SELECT
            p.project_id,
            p.name AS project_name,
            pm.role,
            p.status AS project_status
        FROM project_members pm
        JOIN projects p ON pm.project_id = p.project_id
        WHERE pm.employee_id = ?
        ORDER BY
            CASE pm.role
                WHEN 'lead' THEN 1
                WHEN 'core' THEN 2
                WHEN 'contributor' THEN 3
                ELSE 4
            END
    """
    rows = conn.execute(sql, (employee_id,)).fetchall()
    return _rows_to_list(rows)


def get_projects_by_status(conn: sqlite3.Connection, status: str) -> list[dict[str, Any]]:
    """
    按状态筛选项目。

    Args:
        conn: 数据库连接。
        status: 项目状态，可选值: 'active', 'planning', 'completed', 'on_hold'。

    Returns:
        项目字典列表，每项包含 project_id, name, lead_id, status,
        start_date, end_date, budget。
    """
    sql = """
        SELECT project_id, name, lead_id, status, start_date, end_date, budget
        FROM projects
        WHERE status = ?
        ORDER BY project_id
    """
    rows = conn.execute(sql, (status,)).fetchall()
    return _rows_to_list(rows)


# ── 考勤查询 ──────────────────────────────────────────────────────────


def get_attendance_late_count(
    conn: sqlite3.Connection,
    employee_id: str,
    year: int,
    month: int,
) -> int:
    """
    查询某员工某年某月的迟到次数。

    使用 date LIKE 'YYYY-MM-%' 模式匹配月份，status='late' 筛选迟到记录。

    Args:
        conn: 数据库连接。
        employee_id: 员工工号。
        year: 年份，如 2026。
        month: 月份，如 2。

    Returns:
        迟到次数（整数）。
    """
    date_prefix = f"{year:04d}-{month:02d}-%"
    sql = """
        SELECT COUNT(*) AS cnt
        FROM attendance
        WHERE employee_id = ? AND status = ? AND date LIKE ?
    """
    row = conn.execute(sql, (employee_id, "late", date_prefix)).fetchone()
    return row["cnt"] if row else 0


# ── 绩效查询 ──────────────────────────────────────────────────────────


def get_performance_reviews(
    conn: sqlite3.Connection,
    employee_id: str,
    year: int,
) -> list[dict[str, Any]]:
    """
    查询某员工某年所有季度的绩效数据。

    Args:
        conn: 数据库连接。
        employee_id: 员工工号。
        year: 年份，如 2025。

    Returns:
        绩效字典列表，按 quarter 升序排列。
        每项包含 quarter, kpi_score, grade。
    """
    sql = """
        SELECT quarter, kpi_score, grade
        FROM performance_reviews
        WHERE employee_id = ? AND year = ?
        ORDER BY quarter
    """
    rows = conn.execute(sql, (employee_id, year)).fetchall()
    return _rows_to_list(rows)
