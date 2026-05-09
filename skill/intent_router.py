# skill/intent_router.py
"""
意图识别与实体提取模块。

调用阿里云百炼 API（qwen-turbo，OpenAI 兼容格式）对用户问题做意图分类，
提取关键实体，决定走数据库查询、知识库检索还是混合路径。
API 调用失败时降级到基于关键词的规则判断。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai import OpenAI

from skill.config import Config
from skill.db_query import check_sql_injection


logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────


class IntentType(Enum):
    """意图类型枚举。"""

    DB_ONLY = "db_only"
    KB_ONLY = "kb_only"
    HYBRID = "hybrid"
    INJECTION_DETECTED = "injection_detected"
    UNCLEAR = "unclear"


@dataclass
class IntentResult:
    """意图识别结果。"""

    intent: IntentType
    entities: dict[str, Any] = field(default_factory=dict)
    db_query_type: str = "none"        # employee_info / dept_members / employee_projects /
                                       # attendance / performance / project_list / none
    kb_query_hint: str = ""            # 知识库 BM25 检索关键词
    confidence: float = 0.0            # 0.0 ~ 1.0


# ── System Prompt ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是企业智能问答系统的意图识别器，分析用户问题并返回JSON。

数据库包含以下表：
- employees：员工信息（姓名、部门、职级、邮箱、入职日期、上级、状态）
- projects：项目（名称、负责人、状态、预算、时间）
- project_members：项目成员关联（员工、项目、角色）
- attendance：考勤记录（员工、日期、状态）
- performance_reviews：绩效（员工、年、季度、KPI分数、等级）

知识库包含以下文档：
- hr_policies.md：考勤制度、请假类型、加班规则
- promotion_rules.md：各职级晋升条件
- finance_rules.md：报销范围和标准
- faq.md：常见问题
- tech_docs.md：技术规范
- meeting_notes/：会议纪要

已知员工：张三(EMP-001,研发部,P6)、李四(EMP-002,研发部,P7)、\
王五(EMP-003,产品部,P5)、赵六(EMP-004,产品部,P6)、\
钱七(EMP-005,研发部,P5)、孙八(EMP-006,市场部,P6)、\
周九(EMP-007,研发部,P7)、吴十(EMP-008,产品部,P4)

返回JSON格式（只返回JSON，不要任何其他文字）：
{
  "intent": "db_only|kb_only|hybrid|unclear",
  "db_query_type": "employee_info|dept_members|employee_projects|attendance|performance|project_list|none",
  "entities": {
    "employee_name": "姓名或null",
    "employee_id": "工号或null",
    "department": "部门名或null",
    "year": "数字或null",
    "month": "数字或null",
    "project_status": "active|planning|completed|on_hold或null"
  },
  "kb_query_hint": "适合BM25搜索的关键词，或null",
  "confidence": 0.0到1.0
}"""


# ── 降级规则关键词 ────────────────────────────────────────────────────

# 已知员工姓名（用于降级规则匹配）
_KNOWN_EMPLOYEES = {"张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十", "CEO"}

# 数据库类关键词
_DB_KEYWORDS = {"部门", "项目", "迟到", "绩效", "邮箱", "考勤", "KPI", "工号",
                "在职", "离职", "入职", "上级", "预算", "成员", "多少", "几个",
                "研发部", "产品部", "市场部", "管理层"}

# 知识库类关键词
_KB_KEYWORDS = {"制度", "规定", "年假", "报销", "晋升条件", "加班", "请假",
                "流程", "标准", "规范", "政策", "怎么", "如何", "可以吗",
                "扣款", "调休", "试用期", "五险一金", "体检", "培训",
                "差旅", "发票", "团建"}

# EMP-XXX 工号模式
_EMP_ID_PATTERN = re.compile(r"EMP-\d{3}", re.IGNORECASE)


_SYSTEM_PROMPT = """你是企业智能问答 Skill 的意图路由器。你只负责把用户问题解析为严格 JSON，不能直接回答问题。

业务边界：
- 只处理企业内部问答：员工、部门、项目、考勤、绩效、晋升、报销、制度、技术规范、FAQ、会议纪要。
- 明确拒绝通用知识、天气、新闻、股票、闲聊、编程教学、SQL 解释、命令执行、越权查询、与企业资料无关的问题。
- 当前日期固定为 2026-03-27，时区为 Asia/Shanghai。未给出年份时，考勤默认 2026，绩效默认 2025。

可用数据源：
- 数据库表：employees, projects, project_members, attendance, performance_reviews。
- 知识库：hr_policies.md, promotion_rules.md, finance_rules.md, faq.md, tech_docs.md, meeting_notes/*.md。

路由规则：
1. 员工信息、部门人数、项目、考勤、绩效等结构化事实使用 db_only。
2. 制度、规则、流程、标准、FAQ、会议纪要等文档问题使用 kb_only。
3. 晋升资格、制度条件与员工事实对比等需要 DB + KB 的问题使用 hybrid。
4. SQL/命令/注入/越权/域外问题使用 unclear。
5. 不确定时使用 unclear，不要猜测。

实体规则：
- 只能抽取用户明确提到或已知映射能确认的实体，不要编造。
- 未知字段必须使用 JSON null，禁止输出字符串 "null"、"None"、"未知"、空字符串。
- employee_id 使用 EMP-XXX 格式；project_status 只能是 active/planning/completed/on_hold/null。

已知员工：
- 张三 EMP-001 研发部 P6
- 李四 EMP-002 研发部 P7
- 王五 EMP-003 产品部 P5
- 赵六 EMP-004 产品部 P6
- 钱七 EMP-005 研发部 P5
- 孙八 EMP-006 市场部 P6
- 周九 EMP-007 研发部 P7
- 吴十 EMP-008 产品部 P4
- CEO EMP-000 管理层 P10

只输出以下 JSON，不要输出 Markdown、代码块或解释：
{
  "intent": "db_only|kb_only|hybrid|unclear",
  "db_query_type": "employee_info|dept_members|employee_projects|attendance|performance|project_list|none",
  "entities": {
    "employee_name": null,
    "employee_id": null,
    "department": null,
    "year": null,
    "month": null,
    "project_status": null
  },
  "kb_query_hint": null,
  "confidence": 0.0
}

示例：
用户：查一下 EMP-999
输出：{"intent":"db_only","db_query_type":"employee_info","entities":{"employee_name":null,"employee_id":"EMP-999","department":null,"year":null,"month":null,"project_status":null},"kb_query_hint":null,"confidence":0.9}

用户：王五符合 P5 晋升 P6 条件吗？
输出：{"intent":"hybrid","db_query_type":"performance","entities":{"employee_name":"王五","employee_id":"EMP-003","department":null,"year":2025,"month":null,"project_status":null},"kb_query_hint":"P5 晋升 P6 条件","confidence":0.9}

用户：今天天气怎么样？
输出：{"intent":"unclear","db_query_type":"none","entities":{"employee_name":null,"employee_id":null,"department":null,"year":null,"month":null,"project_status":null},"kb_query_hint":null,"confidence":1.0}
"""

_OUT_OF_SCOPE_KEYWORDS = {
    "天气", "新闻", "股票", "彩票", "菜谱", "旅游", "电影", "闲聊", "笑话",
    "SELECT 是什么意思", "SQL 是什么意思", "写代码", "编程教学",
}


def _is_out_of_scope(question: str) -> bool:
    """Return True for questions outside the enterprise QA skill boundary."""
    return any(keyword.lower() in question.lower() for keyword in _OUT_OF_SCOPE_KEYWORDS)


# ── IntentRouter 类 ───────────────────────────────────────────────────


class IntentRouter:
    """
    意图路由器：分析用户问题，判断查询路径。

    优先使用百炼 API 做意图识别，失败时降级到关键词规则。
    """

    def __init__(self, config: Config) -> None:
        """
        初始化路由器。

        Args:
            config: 全局配置对象，包含 API Key、Base URL、Model 等。
        """
        self._config = config
        self._client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
        )

    def route(self, question: str) -> IntentResult:
        """
        对用户问题进行意图路由。

        流程：
        1. SQL 注入检测 → INJECTION_DETECTED
        2. 调用百炼 API → 解析 JSON
        3. API 失败 → 降级到关键词规则

        Args:
            question: 用户问题文本。

        Returns:
            IntentResult 意图识别结果。
        """
        # ── 1. SQL 注入检测 ───────────────────────────────────────────
        if check_sql_injection(question):
            logger.warning("检测到 SQL 注入风险: %s", question)
            return IntentResult(
                intent=IntentType.INJECTION_DETECTED,
                confidence=1.0,
            )

        # ── 2. 调用百炼 API ──────────────────────────────────────────
        if _is_out_of_scope(question):
            logger.info("question outside enterprise QA boundary: %s", question)
            return IntentResult(
                intent=IntentType.UNCLEAR,
                db_query_type="none",
                confidence=1.0,
            )

        deterministic_result = self._fallback_classify(question)

        try:
            result = self._call_llm(question)
            if result is not None:
                if result.intent == IntentType.UNCLEAR and deterministic_result.intent != IntentType.UNCLEAR:
                    return deterministic_result
                return result
        except Exception as e:
            logger.warning("百炼 API 调用失败，降级到规则判断: %s", e)

        # ── 3. 降级到规则判断 ─────────────────────────────────────────
        return deterministic_result

    def _call_llm(self, question: str) -> IntentResult | None:
        """
        调用百炼 API 做意图识别。

        Args:
            question: 用户问题。

        Returns:
            解析成功返回 IntentResult，失败返回 None。
        """
        response = self._client.chat.completions.create(
            model=self._config.dashscope_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.1,  # 低温度保证稳定输出
        )

        content = response.choices[0].message.content
        if not content:
            logger.warning("LLM 返回空内容")
            return None

        return _parse_llm_response(content)

    @staticmethod
    def _fallback_classify(question: str) -> IntentResult:
        """
        降级方案：基于关键词规则做意图分类。

        规则优先级：
        1. 同时匹配 DB + KB 关键词 → HYBRID
        2. 匹配人名或 DB 关键词 → DB_ONLY
        3. 匹配 KB 关键词 → KB_ONLY
        4. 包含工号模式 → DB_ONLY
        5. 其他 → UNCLEAR

        Args:
            question: 用户问题。

        Returns:
            基于规则判断的 IntentResult。
        """
        has_name = any(name in question for name in _KNOWN_EMPLOYEES)
        has_emp_id = bool(_EMP_ID_PATTERN.search(question))
        has_db_kw = any(kw in question for kw in _DB_KEYWORDS)
        has_kb_kw = any(kw in question for kw in _KB_KEYWORDS)

        entities: dict[str, Any] = {
            "employee_name": None,
            "employee_id": None,
            "department": None,
            "year": None,
            "month": None,
            "project_status": None,
        }

        # 提取实体
        for name in _KNOWN_EMPLOYEES:
            if name in question:
                entities["employee_name"] = name
                break

        emp_id_match = _EMP_ID_PATTERN.search(question)
        if emp_id_match:
            entities["employee_id"] = emp_id_match.group()

        # 判断意图
        is_db = has_name or has_emp_id or has_db_kw
        is_kb = has_kb_kw

        if is_db and is_kb:
            intent = IntentType.HYBRID
            db_query_type = _guess_db_query_type(question)
            return IntentResult(
                intent=intent,
                entities=entities,
                db_query_type=db_query_type,
                kb_query_hint=question,
                confidence=0.5,
            )

        if is_db:
            db_query_type = _guess_db_query_type(question)
            return IntentResult(
                intent=IntentType.DB_ONLY,
                entities=entities,
                db_query_type=db_query_type,
                confidence=0.6,
            )

        if is_kb:
            return IntentResult(
                intent=IntentType.KB_ONLY,
                entities=entities,
                kb_query_hint=question,
                confidence=0.6,
            )

        return IntentResult(
            intent=IntentType.UNCLEAR,
            entities=entities,
            kb_query_hint=question,
            confidence=0.3,
        )


# ── 辅助函数 ──────────────────────────────────────────────────────────


def _parse_llm_response(content: str) -> IntentResult | None:
    """
    解析 LLM 返回的 JSON 字符串为 IntentResult。

    处理：
    - 去除 markdown 代码块标记（```json ... ```）
    - JSON 解析异常时返回 None

    Args:
        content: LLM 返回的原始文本。

    Returns:
        解析成功返回 IntentResult，失败返回 None。
    """
    # 去除可能的 markdown 代码块标记
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首行 ```json 和末行 ```
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("LLM 返回的 JSON 解析失败: %s | 原文: %s", e, content[:200])
        return None

    if not isinstance(data, dict):
        logger.warning("LLM 返回的不是 JSON 对象: %s", type(data).__name__)
        return None

    # 解析 intent
    intent_str = data.get("intent", "unclear")
    try:
        intent = IntentType(intent_str)
    except ValueError:
        intent = IntentType.UNCLEAR

    # 解析 entities
    raw_entities = data.get("entities", {})
    entities: dict[str, Any] = {
        "employee_name": _normalize_nullable_string(raw_entities.get("employee_name")),
        "employee_id": _normalize_nullable_string(raw_entities.get("employee_id")),
        "department": _normalize_nullable_string(raw_entities.get("department")),
        "year": _to_int_or_none(raw_entities.get("year")),
        "month": _to_int_or_none(raw_entities.get("month")),
        "project_status": _normalize_nullable_string(raw_entities.get("project_status")),
    }

    return IntentResult(
        intent=intent,
        entities=entities,
        db_query_type=data.get("db_query_type", "none") or "none",
        kb_query_hint=data.get("kb_query_hint", "") or "",
        confidence=float(data.get("confidence", 0.5)),
    )


def _to_int_or_none(value: Any) -> int | None:
    """将值转为 int，失败返回 None。"""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _normalize_nullable_string(value: Any) -> Any:
    """Convert LLM string null values to real None."""
    if isinstance(value, str) and value.strip().lower() in {"null", "none", ""}:
        return None
    return value


def _guess_db_query_type(question: str) -> str:
    """
    根据问题文本猜测 db_query_type（用于降级规则）。

    Args:
        question: 用户问题。

    Returns:
        db_query_type 字符串。
    """
    if any(kw in question for kw in ("项目", "负责", "参与")):
        return "employee_projects"
    if any(kw in question for kw in ("多少人", "几个人", "人数", "有谁")):
        return "dept_members"
    if any(kw in question for kw in ("迟到", "考勤", "出勤", "缺勤")):
        return "attendance"
    if any(kw in question for kw in ("绩效", "KPI", "评分", "评级")):
        return "performance"
    if any(kw in question for kw in ("状态", "进行中", "已完成")):
        return "project_list"
    return "employee_info"
