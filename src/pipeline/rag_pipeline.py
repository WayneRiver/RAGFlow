"""
RAG 流水线模块

将所有 RAG 组件串联起来，提供端到端的查询能力
"""

from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocess.parser import parse_log_file
from src.preprocess.cleaner import clean_logs
from src.preprocess.chunker import chunk_logs
from src.retrieval.chromadb_retriever import ChromaDBRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import merge_retrieval_results
from src.rerank.bge_rerank import BGEReranker
from src.generator.llm_generator import LLMGenerator

from config import TOP_K, RERANK_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP

class RAGPipeline:
    """
    RAG 完整流水线类
    
    作用：串联检索、重排、生成环节，提供简单的 query() 接口
    
    工作流程：
    用户问题 → 向量检索 + BM25检索 → RRF融合 → BGE重排 → LLM生成 → 返回答案
    """

    def __init__(self):

        print("初始化 RAG Pipeline...")
        self.vector_retriever = ChromaDBRetriever()
        self.bm25_retriever = BM25Retriever()
        # self.reranker = BGEReranker()
        self.llm_generator = LLMGenerator()

    def build_index(self, log_file_path: Path):
        """
        构建知识库索引
        
        参数:
            log_file_path: 日志文件路径
        """

        print("=" * 50)
        print("开始构建索引...")
        print("=" * 50)
        
        # 解析日志
        logs = parse_log_file(log_file_path)
        print(f"   解析到 {len(logs)} 条日志")
        
        # 清洗日志
        cleaned = clean_logs(logs)
        print(f"   清洗后保留 {len(cleaned)} 条日志")
        
        # 分块
        chunks = chunk_logs(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        print(f"   分成 {len(chunks)} 个块")
        
        # 存入向量库
        all_data = self.vector_retriever.collection.get()
        if all_data["ids"]:
            self.vector_retriever.collection.delete(ids=all_data["ids"])
        self.vector_retriever.add_documents(chunks)
        print(f"   向量库已有 {self.vector_retriever.collection.count()} 条文档")
        
        # 构建 BM25 索引
        print("   构建 BM25 索引...")
        self.bm25_retriever.build_index(chunks)
        
        print("=" * 50)
        print("索引构建完成！")
        print("=" * 50)

    def query(self, user_query: str, return_context: bool = False):
        """
        处理用户查询的入口函数
        
        参数:
            user_query: 用户的问题
            return_context: 是否返回检索到的上下文（用于调试）
            
        返回:
            如果 return_context=False: 返回 {"reply": 答案, "usage": token用量}
            如果 return_context=True: 额外返回 {"contexts": 检索到的日志列表}
        """

        # 向量检索
        vector_results = self.vector_retriever.search(
            user_query, 
            top_k=TOP_K
        )

        # BM25 检索
        bm25_results = self.bm25_retriever.search(
            user_query, 
            top_k=TOP_K
        )

        # RRF 融合
        fused_results = merge_retrieval_results(
            vector_results, 
            bm25_results, 
            top_k=TOP_K
        )

        # BGE 重排
        # reranked_results = self.reranker.rerank(user_query, fused_results, top_k=RERANK_TOP_K)
        
        # LLM 生成
        answer = self.llm_generator.generate(
            user_query, 
            fused_results[:RERANK_TOP_K]
        )      

        # 返回结果
        result = {
            "reply": answer["reply"],
            "usage": answer["usage"]
        }
        
        if return_context:
            result["contexts"] = [
                r["document"] for r in fused_results[:RERANK_TOP_K]
            ]
        
        return result

# ===== 测试代码 =====
if __name__ == "__main__":
    from config import RAW_DATA_DIR
    from src.evaluator.ragas_eval import evaluate_rag

    # 创建 pipeline 实例
    pipeline = RAGPipeline()
    
    # 构建索引
    pipeline.build_index(RAW_DATA_DIR / "test.log")
    
    # 测试查询
    test_query = "数据库连接失败是什么原因？"
    
    print(f"\n{'='*60}")
    print(f"👤 用户问题: {test_query}")
    print(f"{'='*60}")
    
    # 查询（返回上下文用于展示）
    result = pipeline.query(test_query, return_context=True)
    
    print(f"\n📚 检索到的上下文:")
    for i, ctx in enumerate(result["contexts"], 1):
        print(f"  {i}. {ctx[:80]}...")
    
    print(f"\n🤖 LLM 回答:")
    print(result["reply"])
    
    print(f"\n📊 Token 用量: {result['usage']}")
