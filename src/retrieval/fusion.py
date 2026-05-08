from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def reciprocal_rank_fusion(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    """
    RRF（Reciprocal Rank Fusion）融合算法
    
    把多个检索结果融合成一个排序列表
    
    公式：score(d) = Σ 1/(k + rank_i(d))
    
    参数:
        results_list: 多个检索结果列表的列表
                      [向量检索结果, BM25检索结果, ...]
        k: 常数，取 60（避免除零）
    
    返回:
        融合后的结果列表
    """

    if not results_list:
        print("没有检索结果需要融合")
        return []

    # 收集所有文档
    doc_scores = {}

    for results in results_list:
        if not results:
            continue

        # 遍历每个文档
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get("id", f"doc_{rank}")
        
            # RRF 分数计算
            score = 1.0 / (k + rank)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "id": doc_id,
                    "score": 0,
                    "document": doc.get("document", ""),
                    "metadata": doc.get("metadata", {}),
                    "sources": []  # 记录来自哪些检索
                }
            doc_scores[doc_id]["score"] += score
            doc_scores[doc_id]["sources"].append(doc.get("source", "unknown"))

    # 按分数降序排序
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    
    return sorted_docs

def merge_retrieval_results(
    chromadb_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    合并 ChromaDB 和 BM25 的检索结果
    
    参数:
        chromadb_results: ChromaDB 向量检索结果
        bm25_results: BM25 关键词检索结果
        top_k: 返回前 K 个结果
    
    返回:
        融合后的结果列表
    """
    # 添加来源标识
    for doc in chromadb_results:
        doc["source"] = "chromadb"
    
    for doc in bm25_results:
        doc["source"] = "bm25"
    
    # RRF 融合
    fused_results = reciprocal_rank_fusion(
        [chromadb_results, bm25_results],
        k=60
    )
    
    # 返回前 top_k 个
    return fused_results[:top_k]

# ===== 测试代码 =====
if __name__ == "__main__":
    # 模拟检索结果
    chromadb_results = [
        {"id": "char_0", "document": "数据库连接失败", "metadata": {"level": "ERROR"}, "score": 0.9},
        {"id": "char_1", "document": "服务启动成功", "metadata": {"level": "INFO"}, "score": 0.7},
        {"id": "char_2", "document": "内存使用率高", "metadata": {"level": "WARNING"}, "score": 0.5},
    ]
    
    bm25_results = [
        {"id": "char_0", "document": "数据库连接失败", "metadata": {"level": "ERROR"}, "score": 0.8},
        {"id": "char_3", "document": "网络超时", "metadata": {"level": "ERROR"}, "score": 0.6},
        {"id": "char_2", "document": "内存使用率高", "metadata": {"level": "WARNING"}, "score": 0.4},
    ]
    
    # 测试融合
    results = merge_retrieval_results(chromadb_results, bm25_results, top_k=3)
    
    print("融合结果:")
    for doc in results:
        print(f"\n{doc['id']}: {doc['document']}")
        print(f"   总分数: {doc['score']:.4f}")
        print(f"   来源: {doc['sources']}")