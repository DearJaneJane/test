# tests/test_config.py
"""
config 模块单元测试。

覆盖场景：
- 缺少必填环境变量时抛出 ConfigError
- 正确读取所有环境变量（含默认值）
- YAML 配置加载（正常 / 缺失字段 / 文件不存在）
"""

import os
import pytest
from unittest.mock import patch

from skill.config import Config, ConfigError, load_config, load_config_from_yaml


# ── 环境变量测试 ──────────────────────────────────────────────────────


class TestLoadConfig:
    """测试 load_config() 函数。"""

    # 最小完整环境变量集合
    _VALID_ENV = {
        "ENTERPRISE_QA_DB_PATH": "./test.db",
        "ENTERPRISE_QA_KB_PATH": "./test_knowledge",
        "DASHSCOPE_API_KEY": "sk-test-key-12345",
    }

    def test_missing_all_required_vars(self):
        """所有必填环境变量缺失 → 抛出 ConfigError 并包含全部缺失项。"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError) as exc_info:
                load_config()

            error_msg = str(exc_info.value)
            assert "ENTERPRISE_QA_DB_PATH" in error_msg
            assert "ENTERPRISE_QA_KB_PATH" in error_msg
            assert "DASHSCOPE_API_KEY" in error_msg

    def test_missing_one_required_var(self):
        """缺少单个必填环境变量 → 抛出 ConfigError 并指出缺失项。"""
        incomplete_env = {
            "ENTERPRISE_QA_DB_PATH": "./test.db",
            "ENTERPRISE_QA_KB_PATH": "./test_knowledge",
            # 缺少 DASHSCOPE_API_KEY
        }
        with patch.dict(os.environ, incomplete_env, clear=True):
            with pytest.raises(ConfigError) as exc_info:
                load_config()

            error_msg = str(exc_info.value)
            assert "DASHSCOPE_API_KEY" in error_msg
            # 已设置的变量不应出现在错误信息中
            assert "ENTERPRISE_QA_DB_PATH: " not in error_msg

    def test_empty_string_treated_as_missing(self):
        """空字符串视为缺失。"""
        env = {**self._VALID_ENV, "DASHSCOPE_API_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError) as exc_info:
                load_config()
            assert "DASHSCOPE_API_KEY" in str(exc_info.value)

    def test_load_with_all_required_vars(self):
        """所有必填变量都设置 → 成功返回 Config 对象。"""
        with patch.dict(os.environ, self._VALID_ENV, clear=True):
            config = load_config()

            assert config.db_path == "./test.db"
            assert config.kb_path == "./test_knowledge"
            assert config.dashscope_api_key == "sk-test-key-12345"

    def test_default_values(self):
        """未设置可选变量时使用默认值。"""
        with patch.dict(os.environ, self._VALID_ENV, clear=True):
            config = load_config()

            assert config.dashscope_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
            assert config.dashscope_model == "qwen-turbo"
            assert config.max_kb_results == 3

    def test_override_optional_vars(self):
        """可选变量可通过环境变量覆盖默认值。"""
        env = {
            **self._VALID_ENV,
            "DASHSCOPE_BASE_URL": "https://custom.api.com/v1",
            "DASHSCOPE_MODEL": "qwen-max",
            "MAX_KB_RESULTS": "5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()

            assert config.dashscope_base_url == "https://custom.api.com/v1"
            assert config.dashscope_model == "qwen-max"
            assert config.max_kb_results == 5

    def test_invalid_max_kb_results(self):
        """MAX_KB_RESULTS 非整数 → 抛出 ConfigError。"""
        env = {**self._VALID_ENV, "MAX_KB_RESULTS": "abc"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="MAX_KB_RESULTS"):
                load_config()

    def test_config_is_frozen(self):
        """Config 对象不可变。"""
        with patch.dict(os.environ, self._VALID_ENV, clear=True):
            config = load_config()
            with pytest.raises(AttributeError):
                config.db_path = "changed"  # type: ignore[misc]


# ── YAML 配置测试 ─────────────────────────────────────────────────────


class TestLoadConfigFromYaml:
    """测试 load_config_from_yaml() 函数。"""

    def test_file_not_found(self, tmp_path):
        """配置文件不存在 → 抛出 ConfigError。"""
        with pytest.raises(ConfigError, match="不存在"):
            load_config_from_yaml(tmp_path / "nonexistent.yaml")

    def test_valid_yaml(self, tmp_path):
        """合法 YAML → 正确解析为 Config。"""
        yaml_content = """
database:
  path: ./enterprise.db
knowledge_base:
  root_path: ./data/knowledge
llm:
  api_key: sk-yaml-test-key
  api_base: https://custom.api.com/v1
  model: qwen-max
search:
  max_results: 5
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = load_config_from_yaml(yaml_file)

        assert config.db_path == "./enterprise.db"
        assert config.kb_path == "./data/knowledge"
        assert config.dashscope_api_key == "sk-yaml-test-key"
        assert config.dashscope_base_url == "https://custom.api.com/v1"
        assert config.dashscope_model == "qwen-max"
        assert config.max_kb_results == 5

    def test_yaml_missing_required_field(self, tmp_path):
        """YAML 缺少必填字段 → 抛出 ConfigError。"""
        yaml_content = """
database:
  path: ./enterprise.db
knowledge_base:
  root_path: ./data/knowledge
llm: {}
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        with pytest.raises(ConfigError, match="llm.api_key"):
            load_config_from_yaml(yaml_file)

    def test_yaml_with_env_ref(self, tmp_path):
        """YAML 中 ${ENV_VAR} 格式能正确解析环境变量。"""
        yaml_content = """
database:
  path: ./enterprise.db
knowledge_base:
  root_path: ./data/knowledge
llm:
  api_key: ${TEST_YAML_API_KEY}
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        with patch.dict(os.environ, {"TEST_YAML_API_KEY": "sk-from-env"}, clear=False):
            config = load_config_from_yaml(yaml_file)
            assert config.dashscope_api_key == "sk-from-env"

    def test_yaml_with_env_ref_default_value(self, tmp_path):
        """YAML 中 ${VAR:-default} 格式：变量未设置时使用默认值。"""
        yaml_content = """
database:
  path: ./enterprise.db
knowledge_base:
  root_path: ./data/knowledge
llm:
  api_key: ${NONEXISTENT_VAR:-sk-default-key}
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        # 确保变量不存在
        env = {k: v for k, v in os.environ.items() if k != "NONEXISTENT_VAR"}
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_yaml(yaml_file)
            assert config.dashscope_api_key == "sk-default-key"

    def test_yaml_invalid_format(self, tmp_path):
        """YAML 格式错误 → 抛出 ConfigError。"""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("::invalid: [yaml", encoding="utf-8")

        with pytest.raises(ConfigError):
            load_config_from_yaml(yaml_file)

    def test_yaml_with_defaults(self, tmp_path):
        """YAML 中省略可选字段时使用默认值。"""
        yaml_content = """
database:
  path: ./enterprise.db
knowledge_base:
  root_path: ./data/knowledge
llm:
  api_key: sk-minimal
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = load_config_from_yaml(yaml_file)

        assert config.dashscope_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert config.dashscope_model == "qwen-turbo"
        assert config.max_kb_results == 3
