from langchain_chroma import Chroma
import os
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document
from rag.splitter import split_knowledge_documents


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"]
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def search_with_score(self, query: str) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k=chroma_conf["k"])

    def _md5_store_path(self) -> str:
        return get_abs_path(chroma_conf["md5_hex_store"])

    def check_md5_hex(self, md5_for_check: str) -> bool:
        store_path = self._md5_store_path()
        if not os.path.exists(store_path):
            open(store_path, "w", encoding="utf-8").close()
            return False

        with open(store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_for_check:
                    return True
        return False

    def save_md5_hex(self, md5_for_check: str):
        with open(self._md5_store_path(), "a", encoding="utf-8") as f:
            f.write(md5_for_check + "\n")

    def _get_file_documents(self, read_path: str) -> list[Document]:
        if read_path.endswith("txt"):
            return txt_loader(read_path)
        if read_path.endswith("pdf"):
            return pdf_loader(read_path)
        return []

    def _split_file(self, documents: list[Document], path: str) -> list[Document]:
        numbered = split_knowledge_documents(documents, path)
        if numbered:
            return numbered
        logger.warning(f"[加载知识库]{path}未识别到编号条目，回退到字符分块")
        return self.splitter.split_documents(documents)

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            if "external" in os.path.normpath(path).split(os.sep):
                logger.info(f"[加载知识库]跳过外部业务数据: {path}")
                continue

            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                logger.warning(f"[加载知识库]{path}无法计算MD5，跳过")
                continue

            if self.check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = self._get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self._split_file(documents, path)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(split_document)
                self.save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 内容加载成功，条目数={len(split_document)}")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败: {str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("开机无反应")
    for r in res:
        print(r.metadata)
        print(r.page_content)
        print("-" * 20)
