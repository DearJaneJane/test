# RAG Design

## Current Implementation

- Loader: recursively loads Markdown files from `data/knowledge`.
- Chunking: splits by level-2 and level-3 Markdown headings.
- Index: jieba tokenization plus BM25.
- Filtering: removes heading-only or very short chunks before returning results.
- Ranking: BM25 score plus small deterministic boosts for section/source matches.
- Answer gate: `QAEngine` filters results below `_MIN_KB_SCORE`.

## Production-Oriented Improvements Already Applied

- Empty-token queries return no KB results.
- Chunks with body text shorter than 20 characters after removing headings are filtered.
- Duplicate `(source_file, section)` results are suppressed.
- Section/source matches receive small, deterministic boosts.
- No-hit answers are explicit and do not fabricate content.

## Recommended Next Improvements

1. Add a hybrid retriever:
   - BM25 for exact policy terms.
   - Vector retrieval for paraphrases.
   - Optional reranker for final top-k ordering.

2. Add metadata:
   - `source_file`, `section`, `doc_title`, `updated_at`, `owner`, `access_level`.
   - Use metadata in answer citations and access control.

3. Add query normalization:
   - Rewrite vague queries into enterprise-domain search hints.
   - Strip noise tokens when the query contains obvious random strings.
   - Preserve unknown tokens for no-hit diagnostics.

4. Add retrieval evaluation:
   - For each golden question, assert expected source file and section.
   - Track top-1 and top-3 recall.
   - Track false-positive rate for nonsense queries.

5. Add observability:
   - Log query, normalized query, top-k sources, scores, selected source, no-hit reason.
   - Avoid logging secrets or sensitive raw fields.
