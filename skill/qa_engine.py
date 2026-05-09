# skill/qa_engine.py
"""
主流程编排模块。

串联 intent_router → db_query / kb_search，组装最终回答。
每个回答末尾标注数据来源，查不到时明确说"未找到"。
"""

import logging
import re
from datetime import date
from typing import Any

from skill.config import Config
from skill.db_query import (
    get_connection,
    get_employee_by_name,
    get_employee_by_id,
    get_department_members,
    get_employee_projects,
    get_attendance_late_count,
    get_performance_reviews,
    get_projects_by_status,
)
from skill.intent_router import IntentResult, IntentRouter, IntentType
from skill.kb_search import KBChunk, KnowledgeBase

logger = logging.getLogger(__name__)

# 测试环境当前日期（README 约定）
_CURRENT_DATE = date(2026, 3, 27)

# KB 结果最低 BM25 分数阈值，低于此值视为无有效匹配
_MIN_KB_SCORE = 0.5


def _is_missing_entity(value: Any) -> bool:
    """Treat LLM string nulls the same as None."""
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none"})


def _missing_employee_label(entities: dict[str, Any], fallback: str) -> str:
    """Choose the clearest label for an employee-not-found message."""
    employee_id = entities.get("employee_id")
    employee_name = entities.get("employee_name")
    if not _is_missing_entity(employee_id):
        return str(employee_id)
    if not _is_missing_entity(employee_name):
        return str(employee_name)
    return fallback


def _kb_not_found_label(question: str) -> str:
    """Prefer the unknown token in mixed queries like 'xyzabc123怎么报销'."""
    match = re.search(r"[A-Za-z0-9_]{6,}", question)
    return match.group(0) if match else question


def _unknown_ascii_core_token(question: str) -> str | None:
    """Extract suspicious ASCII core tokens such as xyzabc123 from a KB query."""
    for token in re.findall(r"\b[A-Za-z0-9_]{6,}\b", question):
        if any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token):
            return token
    return None


def _query_core_token_missing_from_results(question: str, kb_results: list[KBChunk]) -> str | None:
    """Return the unknown core token when retrieved evidence does not mention it."""
    token = _unknown_ascii_core_token(question)
    if not token or not kb_results:
        return None

    top_content = kb_results[0].content.lower()
    return None if token.lower() in top_content else token


class AnswerPolicy:
    """Centralized answer policy for refusal, no-hit, and source requirements."""

    @staticmethod
    def security_rejection() -> str:
        return "检测到潜在安全风险。企业问答系统不处理 SQL、命令、注入或越权类请求。"

    @staticmethod
    def out_of_scope() -> str:
        return (
            "该问题超出企业智能问答 Skill 的范围。我只能回答员工、部门、项目、考勤、绩效、"
            "晋升、报销、制度、技术规范、FAQ 和会议纪要相关问题。"
        )

    @staticmethod
    def employee_not_found(label: str) -> str:
        return f"未找到员工「{label}」的信息，请确认工号或姓名是否正确。"

    @staticmethod
    def kb_not_found(question: str) -> str:
        query = _kb_not_found_label(question)
        if "报销" in question:
            return f"未找到关于「{query}」的相关报销制度，请联系财务部或查阅官方文件。"
        return f"未找到关于「{query}」的相关制度或文档，请联系HR或查阅官方文件。"

    @staticmethod
    def reimbursement_item_not_found(label: str) -> str:
        return f"未找到关于「{label}」的报销规定，如有需要请联系财务部确认。"

    @staticmethod
    def generic_not_found(question: str) -> str:
        return f"未找到关于「{question}」的相关信息。"


class QAEngine:
    """问答引擎：串联意图识别、数据库查询、知识库检索。"""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._conn = get_connection(config.db_path)
        self._kb = KnowledgeBase(config.kb_path)
        self._router = IntentRouter(config)

    def answer(self, question: str) -> str:
        """
        主入口：接收用户问题，返回自然语言回答。

        流程：意图识别 → 数据获取 → 格式化回答 + 来源标注。
        """
        intent = self._router.route(question)

        if intent.intent == IntentType.INJECTION_DETECTED:
            return AnswerPolicy.security_rejection()

        if intent.intent == IntentType.UNCLEAR:
            return AnswerPolicy.out_of_scope()

        if intent.intent == IntentType.INJECTION_DETECTED:
            return "检测到潜在安全风险，该问题无法处理。"

        if intent.intent == IntentType.UNCLEAR:
            return (
                "您的问题比较宽泛，请问您想了解：\n"
                "1. 员工/项目/考勤等具体数据？\n"
                "2. 公司制度/政策/规定？\n"
                "请补充具体问题。"
            )

        db_data: dict[str, Any] | None = None
        kb_results: list[KBChunk] = []

        if intent.intent in (IntentType.DB_ONLY, IntentType.HYBRID):
            db_data = self._handle_db_query(intent)

        if intent.intent in (IntentType.KB_ONLY, IntentType.HYBRID):
            kb_results = self._handle_kb_search(question, intent)

        return self._format_answer(question, intent, db_data, kb_results)

    # ── 数据获取 ──────────────────────────────────────────────────────

    def _resolve_employee(self, entities: dict) -> dict[str, Any] | None:
        """根据 entities 中的 name 或 id 解析员工信息。"""
        name = entities.get("employee_name")
        emp_id = entities.get("employee_id")
        if emp_id:
            return get_employee_by_id(self._conn, emp_id)
        if name:
            return get_employee_by_name(self._conn, name)
        return None

    def _handle_db_query(self, intent: IntentResult) -> dict[str, Any]:
        """根据 db_query_type 分发到对应的查询函数。"""
        qt = intent.db_query_type
        ent = intent.entities
        result: dict[str, Any] = {"type": qt, "found": False, "data": None, "source": ""}

        if qt == "employee_info":
            emp = self._resolve_employee(ent)
            result["data"] = emp
            result["found"] = emp is not None
            result["source"] = f"employees 表 ({emp['employee_id']})" if emp else "employees 表"

        elif qt == "dept_members":
            dept = ent.get("department", "")
            members = get_department_members(self._conn, dept) if dept else []
            result["data"] = members
            result["found"] = len(members) > 0
            result["source"] = "employees 表"

        elif qt == "employee_projects":
            emp = self._resolve_employee(ent)
            if emp:
                projects = get_employee_projects(self._conn, emp["employee_id"])
                result["data"] = {"employee": emp, "projects": projects}
                result["found"] = len(projects) > 0
                result["source"] = "projects 表 + project_members 表"
            else:
                result["data"] = None

        elif qt == "attendance":
            emp = self._resolve_employee(ent)
            if emp:
                year = ent.get("year") or 2026
                month = ent.get("month") or 2
                count = get_attendance_late_count(self._conn, emp["employee_id"], year, month)
                result["data"] = {"employee": emp, "late_count": count, "year": year, "month": month}
                result["found"] = True
                result["source"] = "attendance 表"
            else:
                result["data"] = None

        elif qt == "performance":
            emp = self._resolve_employee(ent)
            if emp:
                year = ent.get("year") or 2025
                reviews = get_performance_reviews(self._conn, emp["employee_id"], year)
                result["data"] = {"employee": emp, "reviews": reviews, "year": year}
                result["found"] = len(reviews) > 0
                result["source"] = "performance_reviews 表"
            else:
                result["data"] = None

        elif qt == "project_list":
            status = ent.get("project_status", "active")
            projects = get_projects_by_status(self._conn, status) if status else []
            result["data"] = projects
            result["found"] = len(projects) > 0
            result["source"] = "projects 表"

        return result

    def _handle_kb_search(self, question: str, intent: IntentResult) -> list[KBChunk]:
        """检索知识库，过滤低分结果。"""
        query = intent.kb_query_hint or question
        top_k = self._config.max_kb_results
        results = self._kb.search(query, top_k=top_k)
        return [r for r in results if r.score >= _MIN_KB_SCORE]

    # ── 格式化回答 ────────────────────────────────────────────────────

    def _format_answer(
        self,
        question: str,
        intent: IntentResult,
        db_data: dict[str, Any] | None,
        kb_results: list[KBChunk],
    ) -> str:
        """根据意图和数据生成最终回答。"""
        if intent.intent == IntentType.HYBRID:
            return self._format_hybrid(question, intent, db_data, kb_results)

        if intent.intent == IntentType.DB_ONLY and db_data:
            return self._format_db(question, intent, db_data)

        if intent.intent == IntentType.KB_ONLY:
            return self._format_kb(question, kb_results)

        return AnswerPolicy.generic_not_found(question)

    def _format_db(self, question: str, intent: IntentResult, db_data: dict) -> str:
        """格式化纯数据库查询结果。"""
        if not db_data.get("found"):
            return AnswerPolicy.employee_not_found(_missing_employee_label(intent.entities, question))

        qt = db_data["type"]
        source = db_data["source"]

        if qt == "employee_info":
            emp = db_data["data"]
            mgr = emp.get("manager_name") or "无"
            lines = [
                f"{emp['name']}，{emp['department']}，职级 {emp['level']}，"
                f"入职日期 {emp['hire_date']}，邮箱 {emp['email']}，"
                f"直属上级：{mgr}。",
                f"\n> 来源：{source}",
            ]
            return "\n".join(lines)

        if qt == "dept_members":
            members = db_data["data"]
            dept = intent.entities.get("department", "")
            names = "、".join(m["name"] for m in members)
            lines = [
                f"{dept}目前有 {len(members)} 名在职员工：{names}。",
                f"\n> 来源：{source}",
            ]
            return "\n".join(lines)

        if qt == "employee_projects":
            info = db_data["data"]
            emp = info["employee"]
            projects = info["projects"]
            lines = [f"{emp['name']}参与了 {len(projects)} 个项目：\n"]
            for p in projects:
                lines.append(f"- **{p['project_name']}**（{p['project_id']}）— 角色：{p['role']}，状态：{p['project_status']}")
            lines.append(f"\n> 来源：{source}")
            return "\n".join(lines)

        if qt == "attendance":
            info = db_data["data"]
            emp = info["employee"]
            lines = [
                f"{emp['name']} {info['year']}年{info['month']}月迟到 {info['late_count']} 次。",
                f"\n> 来源：{source}",
            ]
            return "\n".join(lines)

        if qt == "performance":
            info = db_data["data"]
            emp = info["employee"]
            reviews = info["reviews"]
            if not reviews:
                return f"未找到{emp['name']} {info['year']}年的绩效记录。\n\n> 来源：{source}"
            avg = sum(r["kpi_score"] for r in reviews) / len(reviews)
            lines = [f"{emp['name']} {info['year']}年绩效（共 {len(reviews)} 个季度）：\n"]
            for r in reviews:
                lines.append(f"- Q{r['quarter']}：KPI {r['kpi_score']}，等级 {r['grade']}")
            lines.append(f"\n年度平均 KPI：{avg:.1f}")
            lines.append(f"\n> 来源：{source}")
            return "\n".join(lines)

        if qt == "project_list":
            projects = db_data["data"]
            status = intent.entities.get("project_status", "")
            lines = [f"状态为 {status} 的项目共 {len(projects)} 个：\n"]
            for p in projects:
                lines.append(f"- **{p['name']}**（{p['project_id']}）")
            lines.append(f"\n> 来源：{source}")
            return "\n".join(lines)

        return f"查询结果：{db_data.get('data')}\n\n> 来源：{source}"

    def _format_kb(self, question: str, kb_results: list[KBChunk]) -> str:
        """格式化纯知识库检索结果。"""
        if not kb_results:
            return AnswerPolicy.kb_not_found(question)

        missing_core_token = _query_core_token_missing_from_results(question, kb_results)
        if missing_core_token:
            if "报销" in question:
                return AnswerPolicy.reimbursement_item_not_found(missing_core_token)
            return AnswerPolicy.generic_not_found(missing_core_token)

        if "技术栈" in question:
            tech_chunks = [
                chunk for chunk in kb_results
                if chunk.source_file == "tech_docs.md" and any(section in chunk.section for section in ("后端", "前端"))
            ]
            if tech_chunks:
                lines = ["根据《tech_docs.md》：\n"]
                for chunk in tech_chunks[:2]:
                    lines.append(chunk.content)
                sections = " + ".join(f"tech_docs.md §{chunk.section}" for chunk in tech_chunks[:2])
                lines.append(f"\n> 来源：{sections}")
                return "\n\n".join(lines)

        top = kb_results[0]
        lines = [
            f"根据《{top.source_file}》：\n",
            top.content,
            f"\n> 来源：{top.source_file} §{top.section}",
        ]
        return "\n".join(lines)

    def _format_hybrid(
        self,
        question: str,
        intent: IntentResult,
        db_data: dict[str, Any] | None,
        kb_results: list[KBChunk],
    ) -> str:
        """格式化混合查询结果，含晋升分析。"""
        # 检测是否为晋升分析
        if "晋升" in question and db_data and db_data.get("found"):
            return self._analyze_promotion(intent, db_data, kb_results)

        # 通用混合：拼接 DB + KB 结果
        parts: list[str] = []
        sources: list[str] = []

        if db_data and db_data.get("found"):
            db_answer = self._format_db(question, intent, db_data)
            # 去掉 db_answer 中的来源行，统一在末尾标注
            db_lines = [l for l in db_answer.split("\n") if not l.startswith("> 来源")]
            parts.append("\n".join(db_lines).strip())
            sources.append(db_data.get("source", ""))

        if kb_results:
            top = kb_results[0]
            parts.append(f"根据《{top.source_file}》：\n{top.content}")
            sources.append(f"{top.source_file} §{top.section}")

        if not parts:
            return AnswerPolicy.generic_not_found(question)

        result = "\n\n---\n\n".join(parts)
        source_str = " + ".join(sources)
        return f"{result}\n\n> 来源：{source_str}"

    def _analyze_promotion(
        self,
        intent: IntentResult,
        db_data: dict[str, Any],
        kb_results: list[KBChunk],
    ) -> str:
        """
        晋升分析：对比员工数据与晋升要求，生成 markdown 表格。

        当前硬编码 P5→P6 的判断逻辑（可扩展）。
        """
        emp = self._resolve_employee(intent.entities)
        if not emp:
            name = intent.entities.get("employee_name", "该员工")
            return f'未找到员工「{name}」的信息，无法进行晋升分析。'

        level = emp.get("level", "")
        name = emp["name"]
        emp_id = emp["employee_id"]

        # 获取绩效和项目数据
        reviews = get_performance_reviews(self._conn, emp_id, 2025)
        projects = get_employee_projects(self._conn, emp_id)
        hire_date = date.fromisoformat(emp["hire_date"])
        tenure_days = (_CURRENT_DATE - hire_date).days
        tenure_years = round(tenure_days / 365, 1)

        sources = ["performance_reviews 表", "project_members 表"]

        if level == "P5":
            return self._promotion_p5_to_p6(
                name, level, tenure_years, reviews, projects, kb_results, sources
            )

        # 通用：暂无硬编码规则的职级
        kb_section = kb_results[0].section if kb_results else "晋升标准"
        kb_source = kb_results[0].source_file if kb_results else "promotion_rules.md"
        sources.append(f"{kb_source} §{kb_section}")

        return (
            f"**{name}（{level}）** 的晋升分析暂不支持自动评估，"
            f"请参考《{kb_source}》中的晋升条件。\n\n"
            f"> 来源：{' + '.join(sources)}"
        )

    def _promotion_p5_to_p6(
        self,
        name: str,
        level: str,
        tenure_years: float,
        reviews: list[dict],
        projects: list[dict],
        kb_results: list[KBChunk],
        sources: list[str],
    ) -> str:
        """P5→P6 晋升条件逐条比对。"""
        # ── 条件1：入职年限 ≥ 1 年 ───────────────────────────────────
        tenure_ok = tenure_years >= 1.0
        tenure_status = "✓" if tenure_ok else "✗"

        # ── 条件2：最近连续2季度 KPI ≥ 85 ────────────────────────────
        sorted_reviews = sorted(reviews, key=lambda r: r["quarter"])
        last_two = sorted_reviews[-2:] if len(sorted_reviews) >= 2 else sorted_reviews
        if len(last_two) == 2:
            kpi_ok = all(r["kpi_score"] >= 85 for r in last_two)
            kpi_detail = f"Q{last_two[0]['quarter']}={last_two[0]['kpi_score']}, Q{last_two[1]['quarter']}={last_two[1]['kpi_score']}"
        elif len(last_two) == 1:
            kpi_ok = last_two[0]["kpi_score"] >= 85
            kpi_detail = f"Q{last_two[0]['quarter']}={last_two[0]['kpi_score']}（仅1个季度）"
        else:
            kpi_ok = False
            kpi_detail = "无绩效记录"
        kpi_status = "✓" if kpi_ok else "✗"

        # ── 条件3：项目数 ≥ 3 ────────────────────────────────────────
        proj_count = len(projects)
        proj_ok = proj_count >= 3
        proj_status = "✓" if proj_ok else "✗"

        # ── 综合判断 ─────────────────────────────────────────────────
        all_ok = tenure_ok and kpi_ok and proj_ok
        conclusion = "符合" if all_ok else "不符合"

        # 建议
        suggestions: list[str] = []
        if not kpi_ok:
            suggestions.append("提升 KPI 分数至 85 分以上，保持连续两个季度达标")
        if not proj_ok:
            suggestions.append(f"增加项目参与（当前 {proj_count} 个，需 ≥ 3 个）")

        suggestion_text = ""
        if suggestions:
            suggestion_text = "\n**建议**：\n" + "\n".join(f"- {s}" for s in suggestions)

        # KB 来源
        kb_section = kb_results[0].section if kb_results else "P5 → P6"
        kb_source = kb_results[0].source_file if kb_results else "promotion_rules.md"
        sources.append(f"{kb_source} §{kb_section}")

        return (
            f"**{name}（{level}）晋升分析**\n\n"
            f"| 条件 | 要求 | 当前情况 | 是否满足 |\n"
            f"|------|------|---------|----------|\n"
            f"| 入职年限 | ≥ 1 年 | {tenure_years} 年 | {tenure_status} |\n"
            f"| 连续2季度KPI | ≥ 85 | {kpi_detail} | {kpi_status} |\n"
            f"| 项目数量 | ≥ 3 个 | {proj_count} 个 | {proj_status} |\n\n"
            f"**结论**：{conclusion}晋升条件。{suggestion_text}\n\n"
            f"> 来源：{' + '.join(sources)}"
        )
