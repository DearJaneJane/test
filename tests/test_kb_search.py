# tests/test_kb_search.py
"""
kb_search 模块单元测试。

基于真实知识库文件（data/knowledge/）验证加载、切分和检索效果。
"""

from pathlib import Path

import pytest

from skill.kb_search import KBChunk, KnowledgeBase, load_knowledge_base


# ── Fixtures ──────────────────────────────────────────────────────────

_KB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "knowledge")


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    """模块级 fixture：加载真实知识库并建索引。"""
    return KnowledgeBase(_KB_PATH)


# ── 加载测试 ──────────────────────────────────────────────────────────


class TestLoadKnowledgeBase:
    """测试 load_knowledge_base()。"""

    def test_loads_all_files(self):
        """应加载所有 7 个 md 文件的内容。"""
        chunks = load_knowledge_base(_KB_PATH)
        source_files = {c.source_file for c in chunks}
        assert "hr_policies.md" in source_files
        assert "promotion_rules.md" in source_files
        assert "tech_docs.md" in source_files
        assert "finance_rules.md" in source_files
        assert "faq.md" in source_files

    def test_loads_meeting_notes(self):
        """应递归加载 meeting_notes/ 子目录。"""
        chunks = load_knowledge_base(_KB_PATH)
        source_files = {c.source_file for c in chunks}
        # 路径中包含 meeting_notes 目录
        meeting_sources = [f for f in source_files if "allhands" in f]
        assert len(meeting_sources) >= 1

    def test_chunks_have_sections(self):
        """每个 chunk 都应有非空的 section。"""
        chunks = load_knowledge_base(_KB_PATH)
        for chunk in chunks:
            assert chunk.section, f"chunk from {chunk.source_file} has empty section"

    def test_chunks_have_content(self):
        """每个 chunk 都应有非空的 content。"""
        chunks = load_knowledge_base(_KB_PATH)
        for chunk in chunks:
            assert chunk.content.strip(), f"chunk '{chunk.section}' from {chunk.source_file} has empty content"

    def test_nonexistent_path_raises(self):
        """不存在的路径 → 抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_knowledge_base("/nonexistent/path")


# ── 检索测试 ──────────────────────────────────────────────────────────


class TestKnowledgeBaseSearch:
    """测试 KnowledgeBase.search() 的检索准确性。"""

    def test_search_late_penalty(self, kb: KnowledgeBase):
        """搜索"迟到扣款" → 第一个结果来自 hr_policies.md，章节含"迟到"，内容含"50"。"""
        results = kb.search("迟到扣款", top_k=3)
        assert len(results) > 0

        top = results[0]
        assert top.source_file == "hr_policies.md"
        assert "迟到" in top.section
        assert "50" in top.content

    def test_search_annual_leave(self, kb: KnowledgeBase):
        """搜索"年假天数" → 结果来自 hr_policies.md。"""
        results = kb.search("年假天数", top_k=3)
        assert len(results) > 0

        source_files = [r.source_file for r in results]
        assert "hr_policies.md" in source_files

    def test_search_promotion_p5_to_p6(self, kb: KnowledgeBase):
        """搜索"P5晋升P6" → 结果来自 promotion_rules.md。"""
        results = kb.search("P5晋升P6", top_k=3)
        assert len(results) > 0

        source_files = [r.source_file for r in results]
        assert "promotion_rules.md" in source_files

    def test_search_travel_reimbursement(self, kb: KnowledgeBase):
        """搜索"差旅报销" → 结果来自 finance_rules.md。"""
        results = kb.search("差旅报销", top_k=3)
        assert len(results) > 0

        source_files = [r.source_file for r in results]
        assert "finance_rules.md" in source_files

    def test_search_allhands_meeting(self, kb: KnowledgeBase):
        """搜索"全员大会" → 结果来自 allhands 会议纪要。"""
        results = kb.search("全员大会", top_k=3)
        assert len(results) > 0

        source_files = [r.source_file for r in results]
        has_allhands = any("allhands" in f for f in source_files)
        assert has_allhands, f"Expected allhands in sources, got: {source_files}"

    def test_search_nonsense_low_score(self, kb: KnowledgeBase):
        """搜索无意义字符串 → 所有结果的 score 极低（< 0.1）。"""
        results = kb.search("xyzabc123qqwwee", top_k=3)
        for r in results:
            assert r.score < 0.1, f"Unexpected high score {r.score} for nonsense query"

    def test_search_results_sorted_by_score(self, kb: KnowledgeBase):
        """检索结果按 score 降序排列。"""
        results = kb.search("考勤制度", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_limit(self, kb: KnowledgeBase):
        """top_k 参数限制返回数量。"""
        results_1 = kb.search("制度", top_k=1)
        results_5 = kb.search("制度", top_k=5)
        assert len(results_1) == 1
        assert len(results_5) == 5
