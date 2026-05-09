# skill/config.py
"""
配置管理模块。

从环境变量或 YAML 文件读取配置项，返回不可变的 Config 对象。
必填项缺失时抛出 ConfigError 并给出明确提示。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(Exception):
    """配置错误：环境变量缺失或配置文件无效。"""


@dataclass(frozen=True)
class Config:
    """全局配置对象（不可变）。"""

    db_path: str
    kb_path: str
    dashscope_api_key: str
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-turbo"
    max_kb_results: int = 3


# ── 必填环境变量映射 ──────────────────────────────────────────────────
_REQUIRED_ENV_VARS: dict[str, str] = {
    "ENTERPRISE_QA_DB_PATH": "数据库文件路径（如 ./enterprise.db）",
    "ENTERPRISE_QA_KB_PATH": "知识库目录路径（如 ./data/knowledge）",
    "DASHSCOPE_API_KEY": "阿里云百炼 API Key",
}


def load_config() -> Config:
    """
    从环境变量读取配置。

    必填环境变量：
        - ENTERPRISE_QA_DB_PATH
        - ENTERPRISE_QA_KB_PATH
        - DASHSCOPE_API_KEY

    可选环境变量（有默认值）：
        - DASHSCOPE_BASE_URL
        - DASHSCOPE_MODEL
        - MAX_KB_RESULTS

    Raises:
        ConfigError: 缺少必填环境变量时抛出，包含所有缺失项的提示。
    """
    # ── 校验必填项 ────────────────────────────────────────────────────
    missing: list[str] = []
    for var_name, description in _REQUIRED_ENV_VARS.items():
        if not os.environ.get(var_name):
            missing.append(f"  - {var_name}: {description}")

    if missing:
        details = "\n".join(missing)
        raise ConfigError(
            f"缺少以下必填环境变量:\n{details}\n\n"
            f"请设置后重试，例如:\n"
            f'  export ENTERPRISE_QA_DB_PATH="./enterprise.db"\n'
            f'  export ENTERPRISE_QA_KB_PATH="./data/knowledge"\n'
            f'  export DASHSCOPE_API_KEY="sk-xxx"'
        )

    # ── 读取可选项 ────────────────────────────────────────────────────
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model = os.environ.get("DASHSCOPE_MODEL", "qwen-turbo")

    max_kb_results_raw = os.environ.get("MAX_KB_RESULTS", "3")
    try:
        max_kb_results = int(max_kb_results_raw)
    except ValueError:
        raise ConfigError(
            f"环境变量 MAX_KB_RESULTS 的值无效: '{max_kb_results_raw}'，应为正整数"
        )

    return Config(
        db_path=os.environ["ENTERPRISE_QA_DB_PATH"],
        kb_path=os.environ["ENTERPRISE_QA_KB_PATH"],
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        dashscope_base_url=base_url,
        dashscope_model=model,
        max_kb_results=max_kb_results,
    )


def load_config_from_yaml(path: str | Path) -> Config:
    """
    从 YAML 文件读取配置（备选方案）。

    YAML 结构示例::

        database:
          path: ./enterprise.db
        knowledge_base:
          root_path: ./data/knowledge
        llm:
          api_key: sk-xxx
          api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
          model: qwen-turbo
        search:
          max_results: 3

    Args:
        path: YAML 配置文件路径。

    Raises:
        ConfigError: 文件不存在、格式错误或缺少必填字段时抛出。
    """
    try:
        import yaml
    except ImportError:
        raise ConfigError(
            "使用 YAML 配置需要安装 PyYAML: pip install pyyaml"
        )

    filepath = Path(path)
    if not filepath.exists():
        raise ConfigError(f"配置文件不存在: {filepath}")

    try:
        raw = yaml.safe_load(filepath.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 解析错误: {e}")

    if not isinstance(raw, dict):
        raise ConfigError(f"YAML 配置文件根节点必须是字典，实际为: {type(raw).__name__}")

    # ── 提取字段 ──────────────────────────────────────────────────────
    db_section = raw.get("database", {})
    kb_section = raw.get("knowledge_base", {})
    llm_section = raw.get("llm", {})
    search_section = raw.get("search", {})

    db_path = db_section.get("path")
    kb_path = kb_section.get("root_path")
    api_key = llm_section.get("api_key")

    # 必填字段校验
    missing: list[str] = []
    if not db_path:
        missing.append("  - database.path")
    if not kb_path:
        missing.append("  - knowledge_base.root_path")
    if not api_key:
        missing.append("  - llm.api_key")

    if missing:
        details = "\n".join(missing)
        raise ConfigError(f"YAML 配置缺少以下必填字段:\n{details}")

    # 支持 ${ENV_VAR} 格式的环境变量引用
    api_key = _resolve_env_ref(api_key)

    return Config(
        db_path=db_path,
        kb_path=kb_path,
        dashscope_api_key=api_key,
        dashscope_base_url=llm_section.get(
            "api_base",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        dashscope_model=llm_section.get("model", "qwen-turbo"),
        max_kb_results=int(search_section.get("max_results", 3)),
    )


def _resolve_env_ref(value: str) -> str:
    """
    解析 ${ENV_VAR} 格式的环境变量引用。

    如果 value 形如 ${SOME_VAR}，则从环境变量中读取其值。
    支持 ${VAR:-default} 默认值语法。
    """
    if not (value.startswith("${") and value.endswith("}")):
        return value

    inner = value[2:-1]  # 去掉 ${ 和 }

    # 处理 ${VAR:-default} 格式
    if ":-" in inner:
        var_name, default = inner.split(":-", 1)
        return os.environ.get(var_name, default)

    # 纯 ${VAR} 格式
    env_value = os.environ.get(inner)
    if env_value is None:
        raise ConfigError(
            f"YAML 中引用了环境变量 ${{{inner}}}，但该变量未设置"
        )
    return env_value
