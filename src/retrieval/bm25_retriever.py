from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rank_bm25 import BM25Okapi
import jieba

class BM25Retriever:
    """
    BM25 关键词检索器
    
    BM25 是一种基于关键词的检索算法
    特点：适合精确匹配，不依赖向量
    """

    def __init__(self):
        self.bm25 = None
        self.ids = [] # 文档 ID
        self.documents = [] # 原始文档
        self.doc_metadatas = [] # 文档元数据

    def build_index(self, chunks: list[dict]) -> None:
        """
        构建 BM25 索引
        
        参数:
            chunks: 分块列表，每个元素包含 text、metadata
        """
        if not chunks:
            print("没有文档需要构建索引")
            return

        # 准备文档和元数据
        self.ids = []
        self.documents = []
        self.doc_metadatas = []

        for chunk in chunks:
            self.ids.append(chunk["id"])
            self.documents.append(chunk["text"])
            self.doc_metadatas.append(chunk.get("metadata", {}))

        # 分词（用 jieba 对中文分词）
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"BM25 索引构建完成，共 {len(self.documents)} 个文档")

    def _tokenize(self, text: str) -> list[str]:
        """
        分词
        
        用 jieba 对中文分词，英文按空格分词
        """
        words = jieba.cut(text)
        return list(words)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        检索与查询关键词匹配的文档
        
        参数:
            query: 用户问题
            top_k: 返回前 K 个最相关的文档
        
        返回:
            检索结果列表
        """

        if self.bm25 is None:
            print("请先调用 build_index 构建索引")
            return []

        # 对查询分词
        tokenized_query = self._tokenize(query)

        # 计算每个文档的 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)

        # 按分数排序，取前 top_k 个
        top_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "document": self.documents[idx],
                "metadata": self.doc_metadatas[idx],
                "score": scores[idx]
            })

        return results

# ===== 测试代码 =====
if __name__ == "__main__":
    retriever = BM25Retriever()
    
    # 测试数据
    test_chunks = [
        {"id": "char_0", "text": "数据库连接失败，请检查网络", "metadata": {"time": "2024-01-01 10:00:00", "level": "ERROR"}},
        {"id": "char_1", "text": "服务启动成功", "metadata": {"time": "2024-01-01 10:01:00", "level": "INFO"}},
        {"id": "char_2", "text": "内存使用率超过80%", "metadata": {"time": "2024-01-01 10:02:00", "level": "WARNING"}},
    ]
    
    # 构建索引
    retriever.build_index(test_chunks)
    
    # 检索
    results = retriever.search("内存故障", top_k=2)
    
    print("\n检索结果:")
    for result in results:
        print(f"\n结果 {result['id']}:")
        print(f"  内容: {result['document']}")
        print(f"  分数: {result['score']:.4f}")
        
        
       