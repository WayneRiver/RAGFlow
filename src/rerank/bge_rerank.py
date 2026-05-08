"""
BGE 重排模块

本模块使用 BGE-reranker 对检索结果进行重排序
采用 Cross-encoder 架构，精度高于向量检索
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from FlagEmbedding import FlagReranker

# 导入配置
from config import (
    RERANK_MODEL,      # 重排模型名称
    RERANK_TOP_K      # 重排后返回的数量
)

class BGEReranker:
    """
    BGE 重排器类
    
    作用：对初步检索的结果进行二次精排
    原理：使用 Cross-encoder 架构，同时编码 Query 和 Document，
          计算两者的语义相关性分数
    """

    def __init__(self, model_name: str = None, use_fp16: bool = True):
        """
        初始化重排器
        
        参数:
            model_name: 模型名称，默认从 config 读取
            use_fp16: 是否使用 FP16 加速（需要 GPU）
                      True: 速度快，精度略降
                      False: 速度慢，精度更高
        """

        model_name = model_name or RERANK_MODEL

        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        print(f"BGE 重排器初始化完成: {model_name}")
    
    def rerank(self, query: str, documents: list[dict], top_k: int = None) -> list[dict]:
        """
        对文档列表进行重排序
        
        参数:
            query: 用户的查询（问题）
            documents: 初步检索的结果列表
                      每个元素是字典，包含:
                      - id: 文档 ID
                      - document: 文档内容
                      - metadata: 元数据
            top_k: 返回前 K 个最相关的文档
        
        返回:
            重排后的文档列表
        """

        if not documents:
            print("没有文档需要重排")
            return []
        
        top_k = top_k or RERANK_TOP_K

        # 构造 Query-Document 对
        # FlagReranker  [[query, doc], [query, doc], ...] 格式
        pairs = []
        for doc in documents:
            doc_text = doc.get("document", "")
            pairs.append([query, doc_text])

        # compute_score 会同时编码所有 Query-Document 对
        # normalize=True 表示对分数进行归一化，输出 0-1 之间的值
        # 分数越高，表示相关性越强
        scores = self.reranker.compute_score(pairs, normalize=True)

        for i, doc in enumerate(documents):
            doc["rerank_score"] = scores[i]

        results = sorted(
            documents,
            key=lambda x: x["rerank_score"],
            reverse=True
        )[:top_k]

        for i, doc in enumerate(results):
            doc["rerank_rank"] = i + 1

        print(f"重排完成，从 {len(documents)} 个文档中选出 Top {top_k}")
        return results

# ===== 测试代码 =====
if __name__ == "__main__":
    from src.preprocess.parser import parse_log_file
    from src.preprocess.cleaner import clean_logs
    from src.preprocess.chunker import chunk_logs
    from src.retrieval.chromadb_retriever import ChromaDBRetriever
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.fusion import merge_retrieval_results
    from config import RAW_DATA_DIR

    # ===== 1. 预处理流水线 =====
    log_file = RAW_DATA_DIR / "test.log"
    if not log_file.exists():
        print(f"文件不存在: {log_file}")
        exit(1)

    logs = parse_log_file(log_file)
    print(f"1. 解析到 {len(logs)} 条日志")

    cleaned = clean_logs(logs)
    print(f"2. 清洗后 {len(cleaned)} 条日志")

    chunks = chunk_logs(cleaned, chunk_size=200, overlap=20)
    print(f"3. 分块后 {len(chunks)} 个块")

    # ===== 2. 构建检索索引 =====
    # ChromaDB 向量索引（用阿里百炼 embedding）
    vector_retriever = ChromaDBRetriever()
    # 先清理旧数据
    all_data = vector_retriever.collection.get()
    if all_data["ids"]:
        vector_retriever.collection.delete(ids=all_data["ids"])
    vector_retriever.add_documents(chunks)

    # BM25 关键词索引
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(chunks)

    # ===== 3. 混合检索测试 =====
    print("\n" + "=" * 50)
    print("混合检索 + 重排测试")
    print("=" * 50)

    query = "数据库连接失败，是什么原因？"
    print(f"\n查询: {query}")

    # ChromaDB 检索
    vector_results = vector_retriever.search(query, top_k=5)
    print(f"向量检索到 {len(vector_results)} 条")

    # BM25 检索
    bm25_results = bm25_retriever.search(query, top_k=5)
    print(f"BM25 检索到 {len(bm25_results)} 条")

    # RRF 融合
    fused_results = merge_retrieval_results(vector_results, bm25_results, top_k=5)
    print(f"RRF 融合后 {len(fused_results)} 条")
    print(f"\n--- 融合结果（重排前）---")
    for doc in fused_results:
        print(f"  [{doc['metadata']['level']}] {doc['document'][:60]}")
        print(f"  RRF 分数: {doc['score']:.4f}  来源: {doc['sources']}")

    # ===== 4. BGE 重排 =====
    reranker = BGEReranker()
    reranked_results = reranker.rerank(query, fused_results, top_k=3)

    print(f"\n--- 重排后 Top 3 ---")
    for doc in reranked_results:
        print(f"  [{doc['metadata']['level']}] {doc['document'][:60]}")
        print(f"  重排分数: {doc['rerank_score']:.4f}  排名: {doc['rerank_rank']}")

    # ===== 5. 对比验证 =====
    print("\n" + "=" * 50)
    print("效果对比：重排前后顺序变化")
    print("=" * 50)
    print(f"{'排名':<6} {'重排前':<40} {'重排后'}")
    print("-" * 85)
    for i in range(max(len(fused_results), len(reranked_results))):
        before = fused_results[i]["document"][:22] if i < len(fused_results) else "-"
        after = reranked_results[i]["document"][:22] if i < len(reranked_results) else "-"
        print(f"{i+1:<6} {before:<25}\t\t {after}")
