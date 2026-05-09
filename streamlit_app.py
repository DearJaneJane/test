"""Streamlit UI for the enterprise QA skill.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from skill.config import ConfigError, load_config_from_yaml
from skill.observability import setup_logging
from skill.qa_engine import QAEngine


st.set_page_config(
    page_title="企业智能问答 Skill",
    page_icon="🏢",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _load_engine() -> QAEngine:
    setup_logging()
    config = load_config_from_yaml("config.yaml")
    return QAEngine(config, enable_memory=True)


def _ensure_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _ask(question: str) -> None:
    engine = _load_engine()
    answer = engine.answer(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    _ensure_history()

    with st.sidebar:
        st.title("企业 QA Skill")
        st.caption("SQLite + Markdown RAG + LLM Router")
        st.divider()
        st.write("能力范围")
        st.markdown(
            "- 员工、部门、项目、考勤、绩效\n"
            "- 晋升、报销、人事制度、技术规范\n"
            "- FAQ、会议纪要\n"
            "- 域外与安全问题严格拒绝"
        )
        st.divider()
        if st.button("清空当前对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("企业智能问答 Skill")
    st.caption("支持带来源引用的企业内部问答；交互模式保留最近 5 轮上下文。")

    examples = [
        "张三的部门是什么",
        "李四的上级是谁",
        "年假怎么算",
        "差旅费报销标准是什么",
        "王五符合P5晋升P6条件吗",
        "xyzabc123 怎么报销",
    ]

    cols = st.columns(3)
    for index, example in enumerate(examples):
        with cols[index % 3]:
            if st.button(example, use_container_width=True):
                try:
                    _ask(example)
                except ConfigError as exc:
                    st.error(str(exc))
                st.rerun()

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("输入企业数据、制度、报销、会议纪要相关问题")
    if question:
        try:
            _ask(question)
        except ConfigError as exc:
            st.error(str(exc))
        st.rerun()


if __name__ == "__main__":
    main()
