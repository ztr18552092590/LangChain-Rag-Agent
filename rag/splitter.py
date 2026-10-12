import os
import re
from langchain_core.documents import Document

ITEM_START = re.compile(r"(?m)^\d+\.\s+")


def infer_category(filepath: str) -> str:
    name = os.path.basename(filepath)
    if "故障" in name:
        return "故障排除"
    if "选购" in name:
        return "选购指南"
    if "保养" in name or "维护" in name:
        return "维护保养"
    if "100问" in name:
        return "常见问答"
    return "其他"


def split_numbered_text(text: str) -> list[str]:
    """按「1. 2. 3.」切成完整知识条目，题干和答案绑在一起。"""
    matches = list(ITEM_START.finditer(text or ""))
    if len(matches) < 3:
        return []

    chunks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[match.start():end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def split_knowledge_documents(documents: list[Document], filepath: str) -> list[Document]:
    text = "\n".join((doc.page_content or "") for doc in documents).strip()
    category = infer_category(filepath)
    source = os.path.basename(filepath)
    items = split_numbered_text(text)
    if not items:
        return []

    return [
        Document(
            page_content=item,
            metadata={
                "source": source,
                "category": category,
                "item_index": i + 1,
            },
        )
        for i, item in enumerate(items)
    ]
