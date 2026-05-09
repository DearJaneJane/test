# main.py
"""
企业智能问答助手 — 命令行入口。

用法：
    python main.py "张三的部门是什么"       # 单次提问
    python main.py --interactive             # 交互模式
    python main.py --init-db                 # 初始化数据库
"""

import argparse
import sys
from pathlib import Path

from skill.config import Config, ConfigError, load_config, load_config_from_yaml
from skill.observability import setup_logging
from skill.qa_engine import QAEngine


_WELCOME = """
╔══════════════════════════════════════════════╗
║         企业智能问答助手 v1.0                ║
║                                              ║
║  支持查询：员工/项目/考勤/绩效/制度/政策     ║
║  输入"退出"或"exit"结束对话                  ║
╚══════════════════════════════════════════════╝
"""

_SEPARATOR = "\n" + "─" * 50 + "\n"


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_config_or_exit(config_path: str | None = None) -> Config:
    """加载配置，失败时打印友好错误信息并退出。"""
    try:
        if config_path:
            return load_config_from_yaml(config_path)
        if _DEFAULT_CONFIG_PATH.exists():
            return load_config_from_yaml(_DEFAULT_CONFIG_PATH)
        return load_config()
    except ConfigError as e:
        print(f"\n❌ 配置错误：\n{e}\n", file=sys.stderr)
        sys.exit(1)


def _run_once(engine: QAEngine, question: str) -> None:
    """单次提问模式。"""
    answer = engine.answer(question)
    print(answer)


def _run_interactive(engine: QAEngine) -> None:
    """交互模式：循环接收问题，输入"退出"或"exit"结束。"""
    print(_WELCOME)

    while True:
        try:
            question = input("📝 请输入您的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not question:
            continue

        if question.lower() in ("退出", "exit", "quit", "q"):
            print("👋 再见！")
            break

        answer = engine.answer(question)
        print(f"\n💡 {answer}")
        print(_SEPARATOR)


def _init_db() -> None:
    """调用 init_db 脚本初始化数据库。"""
    from init_db import init_database, verify_data, _get_db_path

    db_path = _get_db_path()
    init_database(db_path)
    verify_data(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="企业智能问答助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            '  python main.py "张三的部门是什么"\n'
            "  python main.py --interactive\n"
            "  python main.py --init-db"
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="要提问的问题（单次模式）",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="启动交互模式",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="初始化数据库（执行 schema.sql + seed_data.sql）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. Defaults to ./config.yaml when it exists.",
    )
    args = parser.parse_args()

    # 初始化数据库（不需要 API Key）
    if args.init_db:
        _init_db()
        return

    # 无参数时显示帮助
    if not args.question and not args.interactive:
        parser.print_help()
        sys.exit(0)

    # 加载配置并创建引擎
    setup_logging()
    config = _load_config_or_exit(args.config)
    engine = QAEngine(config, enable_memory=args.interactive)

    if args.interactive:
        _run_interactive(engine)
    else:
        _run_once(engine, args.question)


if __name__ == "__main__":
    main()
