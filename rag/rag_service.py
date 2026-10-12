"""检索服务：只返回向量库原文，不再二次总结，避免幻觉。"""

from langchain_core.documents import Document
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf

UNCOVERED_MSG = "【知识库未覆盖】向量库中没有与该问题足够相关的资料。"


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()

    def retriever_docs(self, query: str) -> list[Document]:
        scored = self.vector_store.search_with_score(query)
        max_distance = chroma_conf.get("max_distance", 1.5)
        return [doc for doc, score in scored if score <= max_distance]

    def rag_summarize(self, query: str) -> str:
        scored = self.vector_store.search_with_score(query)
        max_distance = chroma_conf.get("max_distance", 1.5)
        kept = [(doc, score) for doc, score in scored if score <= max_distance]

        if not kept:
            return UNCOVERED_MSG

        lines = []
        for index, (doc, score) in enumerate(kept, start=1):
            category = (doc.metadata or {}).get("category", "未分类")
            source = (doc.metadata or {}).get("source", "")
            lines.append(
                f"【资料{index}｜{category}｜{source}】\n{doc.page_content.strip()}"
            )
        return "\n\n".join(lines)


if __name__ == '__main__':
    rag = RagSummarizeService()
    result = rag.rag_summarize("小户型适合哪些扫地机器人")
    print(result)
