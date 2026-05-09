# skill/kb_search.py
"""
知识库 BM25 检索模块。

加载 Markdown 知识库文件，按标题层级切分为文档片段，
使用 jieba 分词 + BM25 算法提供语义检索能力。
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
class KBChunk:
    """知识库文档切片。"""

    source_file: str   # 文件名，如 hr_policies.md
    section: str       # 所在章节标题，如 "迟到规则"
    content: str       # 段落文本（含标题行）
    score: float = 0.0 # BM25 相关性分数


# ── 停用词（高频无意义词）────────────────────────────────────────────

_STOP_WORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
    "着", "没有", "看", "好", "自己", "这", "他", "她", "它", "们",
    "那", "些", "什么", "怎么", "如何", "可以", "吗", "呢", "吧",
    "啊", "哪", "哪些", "为什么", "多少", "几", "请问",
    "\n", "\r", " ", "\t", "",
    # 标点
    "，", "。", "、", "：", "；", "！", "？", "（", "）",
    "—", """, """, "'", "'", "【", "】", "·",
    "|", "-", "+", "=", "*", "/", "\\", "#",
}


# ── 文档加载与切分 ────────────────────────────────────────────────────


def _split_by_headings(text: str, source_file: str) -> list[KBChunk]:
    """
    按 Markdown 标题（## 或 ###）切分文档为多个 KBChunk。

    切分规则：
    - 以 ## 或 ### 开头的行作为切块边界
    - 每块包含标题行 + 其下所有内容，直到下一个同级或更高级标题
    - # 一级标题不作为切块边界，但记录为上下文

    Args:
        text: Markdown 文件全文。
        source_file: 文件名（不含目录路径）。

    Returns:
        KBChunk 列表。
    """
    chunks: list[KBChunk] = []
    lines = text.split("\n")

    current_section = ""
    current_lines: list[str] = []
    doc_title = ""  # 一级标题，用于丰富 section 上下文

    for line in lines:
        stripped = line.strip()

        # 一级标题：记录但不切块
        if re.match(r"^#\s+", stripped) and not re.match(r"^#{2,}", stripped):
            doc_title = stripped.lstrip("# ").strip()
            continue

        # 二级或三级标题：产生新切块
        if re.match(r"^#{2,3}\s+", stripped):
            # 保存上一个块
            if current_lines and current_section:
                content = "\n".join(current_lines).strip()
                if content:
                    full_section = f"{doc_title} > {current_section}" if doc_title else current_section
                    chunks.append(KBChunk(
                        source_file=source_file,
                        section=full_section,
                        content=content,
                    ))

            current_section = stripped.lstrip("# ").strip()
            current_lines = [line]
            continue

        current_lines.append(line)

    # 保存最后一个块
    if current_lines and current_section:
        content = "\n".join(current_lines).strip()
        if content:
            full_section = f"{doc_title} > {current_section}" if doc_title else current_section
            chunks.append(KBChunk(
                source_file=source_file,
                section=full_section,
                content=content,
            ))

    # 兜底：如果没有任何 ## / ### 标题，则整个文件作为一个块
    if not chunks and text.strip():
        chunks.append(KBChunk(
            source_file=source_file,
            section=doc_title or source_file,
            content=text.strip(),
        ))

    return chunks


def load_knowledge_base(kb_path: str) -> list[KBChunk]:
    """
    加载知识库目录下所有 .md 文件，按标题切分为 KBChunk 列表。

    递归扫描目录（含 meeting_notes/ 等子目录）。

    Args:
        kb_path: 知识库根目录路径。

    Returns:
        所有文档切片的列表。

    Raises:
        FileNotFoundError: 知识库目录不存在。
    """
    root = Path(kb_path)
    if not root.exists():
        raise FileNotFoundError(f"知识库目录不存在: {kb_path}")

    chunks: list[KBChunk] = []

    for md_file in sorted(root.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        # source_file 保留相对路径（如 meeting_notes/2026-03-01-allhands.md）
        relative = md_file.relative_to(root)
        file_chunks = _split_by_headings(text, str(relative))
        chunks.extend(file_chunks)

    return chunks


# ── 分词与索引 ────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """
    使用 jieba 对中文文本分词，过滤停用词和单字符 token。

    Args:
        text: 待分词文本。

    Returns:
        分词结果列表（已去停用词）。
    """
    words = jieba.lcut(text)
    return [w for w in words if w not in _STOP_WORDS and len(w.strip()) > 0]


def build_index(chunks: list[KBChunk]) -> BM25Okapi:
    """
    对 KBChunk 列表构建 BM25 索引。

    对每个 chunk 的 content + section 拼接后分词，作为文档表示。

    Args:
        chunks: 文档切片列表。

    Returns:
        BM25Okapi 索引对象。
    """
    corpus = [_tokenize(f"{chunk.section} {chunk.content}") for chunk in chunks]
    return BM25Okapi(corpus)


def search(
    query: str,
    chunks: list[KBChunk],
    index: BM25Okapi,
    top_k: int = 3,
) -> list[KBChunk]:
    """
    基于 BM25 检索最相关的 top_k 个文档片段。

    Args:
        query: 用户查询文本。
        chunks: 文档切片列表（与 build_index 时顺序一致）。
        index: BM25 索引。
        top_k: 返回的最大结果数。

    Returns:
        按 score 降序排列的 KBChunk 列表，score 已填充。
    """
    query_tokens = _tokenize(query)
    scores = index.get_scores(query_tokens)

    # 将 score 关联到 chunk，按分数降序取 top_k
    scored_chunks: list[KBChunk] = []
    for i, score_val in enumerate(scores):
        chunk = chunks[i]
        scored_chunks.append(KBChunk(
            source_file=chunk.source_file,
            section=chunk.section,
            content=chunk.content,
            score=float(score_val),
        ))

    scored_chunks.sort(key=lambda c: c.score, reverse=True)
    return scored_chunks[:top_k]


# ── 封装类 ────────────────────────────────────────────────────────────


class KnowledgeBase:
    """
    知识库管理器：封装加载、索引、检索的完整流程。

    初始化时一次性加载并建索引，后续 search 调用复用索引。
    """

    def __init__(self, kb_path: str) -> None:
        """
        加载知识库并构建 BM25 索引。

        Args:
            kb_path: 知识库根目录路径。

        Raises:
            FileNotFoundError: 目录不存在。
        """
        self._chunks = load_knowledge_base(kb_path)
        self._index = build_index(self._chunks) if self._chunks else None

    @property
    def chunks(self) -> list[KBChunk]:
        """返回所有文档切片（只读）。"""
        return self._chunks

    def search(self, query: str, top_k: int = 3) -> list[KBChunk]:
        """
        检索与 query 最相关的 top_k 个文档片段。

        Args:
            query: 用户查询文本。
            top_k: 返回的最大结果数。

        Returns:
            按 score 降序排列的 KBChunk 列表。
            索引为空时返回空列表。
        """
        if not self._index or not self._chunks:
            return []
        return search(query, self._chunks, self._index, top_k)
