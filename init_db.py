# init_db.py
"""
企业智能问答助手 — 数据库初始化脚本。

读取 data/schema.sql 建表，读取 data/seed_data.sql 导入种子数据。
数据库路径从环境变量 ENTERPRISE_QA_DB_PATH 读取，默认值 ./enterprise.db。

用法:
    python init_db.py            # 交互式初始化
    python init_db.py --force    # 强制覆盖，不询问
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path


# ── 路径常量 ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR / "data"
_SCHEMA_PATH = _DATA_DIR / "schema.sql"
_SEED_PATH = _DATA_DIR / "seed_data.sql"
_CONFIG_PATH = _SCRIPT_DIR / "config.yaml"

# 验证期望值
_EXPECTED_COUNTS = {
    "employees": 10,
    "projects": 5,
    "project_members": 12,
    "attendance": 40,
    "performance_reviews": 17,
}

_EXPECTED_ACTIVE_EMPLOYEES = 9


def _get_db_path() -> Path:
    """从环境变量读取数据库路径，未设置时使用默认值。"""
    raw = _get_db_path_from_yaml() or os.environ.get("ENTERPRISE_QA_DB_PATH") or "./enterprise.db"
    return Path(raw).resolve()


def _get_db_path_from_yaml() -> str | None:
    """Read database.path from config.yaml when present."""
    if not _CONFIG_PATH.exists():
        return None

    try:
        import yaml
    except ImportError:
        return None

    try:
        raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    database = raw.get("database")
    if not isinstance(database, dict):
        return None

    path = database.get("path")
    return str(path) if path else None


def _read_sql_file(path: Path) -> str:
    """读取 SQL 文件内容，文件不存在时报错退出。"""
    if not path.exists():
        print(f"✗ SQL 文件不存在: {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def _confirm_overwrite(db_path: Path) -> bool:
    """数据库已存在时询问用户是否覆盖。"""
    while True:
        answer = input(f"⚠ 数据库已存在: {db_path}\n  是否覆盖？(y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  请输入 y 或 n")


def init_database(db_path: Path, *, force: bool = False) -> None:
    """
    初始化数据库：建表 + 导入种子数据。

    Args:
        db_path: 数据库文件路径。
        force: 为 True 时跳过覆盖确认。
    """
    # ── 1. 处理已有数据库 ─────────────────────────────────────────────
    if db_path.exists():
        if not force and not _confirm_overwrite(db_path):
            print("✗ 已取消初始化。")
            sys.exit(0)
        db_path.unlink()
        print(f"✓ 已删除旧数据库: {db_path}")

    # ── 2. 读取 SQL 文件 ──────────────────────────────────────────────
    schema_sql = _read_sql_file(_SCHEMA_PATH)
    seed_sql = _read_sql_file(_SEED_PATH)

    # ── 3. 执行建表 + 导入数据 ────────────────────────────────────────
    # 确保父目录存在
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(schema_sql)
        print("✓ 表结构创建完成")

        conn.executescript(seed_sql)
        print("✓ 种子数据导入完成")
    except sqlite3.Error as e:
        print(f"✗ 数据库初始化失败: {e}")
        conn.close()
        # 清理失败的数据库文件
        if db_path.exists():
            db_path.unlink()
        sys.exit(1)
    finally:
        conn.close()

    print(f"\n✓ 数据库初始化完成: {db_path}")


def verify_data(db_path: Path) -> bool:
    """
    验证数据完整性，打印各表记录数并与期望值比对。

    Returns:
        所有验证通过返回 True，否则 False。
    """
    if not db_path.exists():
        print(f"✗ 数据库文件不存在: {db_path}")
        return False

    conn = sqlite3.connect(str(db_path))
    all_passed = True

    print("\n" + "=" * 50)
    print("  数据验证")
    print("=" * 50)

    try:
        # 各表记录数验证
        for table, expected in _EXPECTED_COUNTS.items():
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 — 表名为硬编码常量，非用户输入
            actual = cursor.fetchone()[0]
            status = "✓" if actual == expected else "✗"
            if actual != expected:
                all_passed = False
            print(f"  {status} {table}: {actual} 条 (预期 {expected})")

        # 在职员工数验证
        cursor = conn.execute(
            "SELECT COUNT(*) FROM employees WHERE status = ?", ("active",)
        )
        active_count = cursor.fetchone()[0]
        status = "✓" if active_count == _EXPECTED_ACTIVE_EMPLOYEES else "✗"
        if active_count != _EXPECTED_ACTIVE_EMPLOYEES:
            all_passed = False
        print(f"  {status} 在职员工数: {active_count} 人 (预期 {_EXPECTED_ACTIVE_EMPLOYEES})")

        # 快速抽检
        print("\n" + "=" * 50)
        print("  快速抽检")
        print("=" * 50)

        cursor = conn.execute(
            "SELECT department FROM employees WHERE employee_id = ?", ("EMP-001",)
        )
        row = cursor.fetchone()
        dept = row[0] if row else "未找到"
        status = "✓" if dept == "研发部" else "✗"
        print(f"  {status} 张三的部门: {dept} (预期 研发部)")

        cursor = conn.execute(
            "SELECT COUNT(*) FROM employees WHERE department = ? AND status = ?",
            ("研发部", "active"),
        )
        rd_count = cursor.fetchone()[0]
        status = "✓" if rd_count == 4 else "✗"
        print(f"  {status} 研发部在职人数: {rd_count} (预期 4)")

        cursor = conn.execute(
            "SELECT COUNT(*) FROM attendance WHERE employee_id = ? AND status = ? AND date LIKE ?",
            ("EMP-001", "late", "2026-02-%"),
        )
        late_count = cursor.fetchone()[0]
        status = "✓" if late_count == 2 else "✗"
        print(f"  {status} 张三 2 月迟到次数: {late_count} (预期 2)")

        cursor = conn.execute(
            "SELECT AVG(kpi_score) FROM performance_reviews WHERE employee_id = ?",
            ("EMP-003",),
        )
        avg_kpi = cursor.fetchone()[0]
        avg_kpi_rounded = round(avg_kpi, 1) if avg_kpi is not None else None
        status = "✓" if avg_kpi_rounded == 80.0 else "✗"
        print(f"  {status} 王五平均 KPI: {avg_kpi_rounded} (预期 80.0)")

    except sqlite3.Error as e:
        print(f"✗ 验证过程出错: {e}")
        all_passed = False
    finally:
        conn.close()

    print("\n" + "=" * 50)
    if all_passed:
        print("  ✓ 全部验证通过！")
    else:
        print("  ✗ 存在验证失败项，请检查数据。")
    print("=" * 50)

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="企业智能问答助手 — 数据库初始化")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制覆盖已有数据库，不询问确认",
    )
    args = parser.parse_args()

    db_path = _get_db_path()

    print("=" * 50)
    print("  企业智能问答助手 — 数据库初始化")
    print("=" * 50)
    print(f"  数据库路径: {db_path}")
    print(f"  Schema:     {_SCHEMA_PATH}")
    print(f"  Seed Data:  {_SEED_PATH}")
    print("=" * 50 + "\n")

    init_database(db_path, force=args.force)
    verify_data(db_path)


if __name__ == "__main__":
    main()
