"""
LLM 故障分析模块

将检索到的日志 + 用户问题输入 qwen-plus，生成自然语言的故障分析报告。
支持多轮对话历史。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dashscope
from dashscope import Generation
from http import HTTPStatus

from config import DASHSCOPE_API_KEY, LLM_MODEL

# 设置阿里云百炼 API Key
dashscope.api_key = DASHSCOPE_API_KEY

class LLMGenerator:
    """
    LLM 故障分析器

    功能：
    1. 接收重排后的日志 + 用户问题
    2. 调用 千问模型 生成诊断分析
    3. 支持多轮对话（传入 history）
    """

    def __init__(self, model: str = None):
        self.model = model or LLM_MODEL

    def generate(
        self,
        query: str,
        context_docs: list[dict],
        history: list[dict] = None,
    ) -> dict:
        """
        分析故障

        参数:
            query: 用户问题，如 "数据库为什么连不上了？"
            context_docs: 重排后的日志列表，每个元素含 document、metadata
            history: 多轮对话历史，格式同 dashscope messages
                     例：[{"role": "assistant", "content": "..."}]

        返回:
            { "reply": "LLM 回复全文", "usage": {总 token 数} }
        """

        prompt = self._build_prompt(query, context_docs)

        messages = [{"role": "system", "content": self._system_prompt()}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        response = Generation.call(
            model=self.model,
            messages=messages,
            result_format="message",
            temperature=0.3
        )

        # 检查响应状态
        if response.status_code != HTTPStatus.OK:
            return {
                "reply": "",
                "success": False,
                "error": f"调用失败: {response.code} - {response.message}"
            }
        
        # 提取生成的文本
        reply = response.output.choices[0].message.content
        usage = response.usage
        
        return {
            "reply": reply,
            "usage": usage,
            "success": True,
            "error": None,
        }

    def _system_prompt(self) -> str:
        """系统提示词：定义 AI 的角色和分析框架"""
        return (
            "你是一个专业的运维故障分析专家。\n\n"
            "你的任务是根据运维日志和用户的问题，分析故障原因并给出修复建议。\n\n"
            "要求：\n"
            "1. 只基于提供的日志内容分析，不要凭空推测\n"
            "2. 先指出故障根因，再给修复步骤\n"
            "3. 按严重程度标注：致命 / 严重 / 警告 / 提示\n"
            "4. 回答要简洁，避免空话\n"
            "5. 回复字数控制在 200 字以内\n"
    )

    def _build_prompt(self, query: str, context_docs: list[dict]) -> str:
        """构建用户提示词"""
        
        context_lines = []
        for i, doc in enumerate(context_docs, start=1):
            level = doc.get("metadata", {}).get("level", "UNKNOWN")
            time = doc.get("metadata", {}).get("time", "")
            message = doc.get("document", "")
            context_lines.append(f"{i}. [{level}] {time} - {message}")
        
        context = "\n".join(context_lines)

        prompt = (
            f"## 角色\n你是一个专业的运维工程师，擅长分析日志定位故障原因。\n\n"
            f"## 规则\n你必须严格基于以下提供的日志内容回答问题，"
            f"不要添加日志中没有的信息。如果日志无法回答问题，请如实说明。\n\n"
            f"## 用户问题\n{query}\n\n"
            f"## 相关日志（必须基于这些内容回答）\n{context}\n\n"
            f"请分析故障原因并给出修复建议。"
        )
        return prompt

# ===== 测试代码 =====
if __name__ == "__main__":
    from src.preprocess.parser import parse_log_file
    from src.preprocess.cleaner import clean_logs
    from src.preprocess.chunker import chunk_logs
    from src.retrieval.chromadb_retriever import ChromaDBRetriever
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.fusion import merge_retrieval_results
    from src.rerank.bge_rerank import BGEReranker
    from config import RAW_DATA_DIR

    # ===== 1. 预处理 =====
    logs = parse_log_file(RAW_DATA_DIR / "test.log")
    cleaned = clean_logs(logs)
    chunks = chunk_logs(cleaned, chunk_size=200, overlap=20)

    # ===== 2. 建索引 =====
    vector_retriever = ChromaDBRetriever()
    all_data = vector_retriever.collection.get()
    if all_data["ids"]:
        vector_retriever.collection.delete(ids=all_data["ids"])
    vector_retriever.add_documents(chunks)

    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(chunks)

    # ===== 3. 检索 + 融合 + 重排 =====
    query = "数据库连接失败，系统无法访问"
    vector_results = vector_retriever.search(query, top_k=5)
    bm25_results = bm25_retriever.search(query, top_k=5)
    fused = merge_retrieval_results(vector_results, bm25_results, top_k=5)

    reranker = BGEReranker()
    top_logs = reranker.rerank(query, fused, top_k=3)

    # ===== 4. 多轮对话测试 =====
    generator = LLMGenerator()

    # 多轮对话：每轮追问一个角度，携带历史
    queries = [
        "数据库连接失败，系统无法访问，什么原因？",
        "具体怎么修复？给操作命令",
        "这些命令对生产环境有什么风险吗？",
    ]

    history = None
    for i, q in enumerate(queries, 1):
        result = generator.generate(q, top_logs, history=history)

        print(f"\n{'=' * 60}")
        print(f"👤 第 {i} 轮: {q}")
        print(f"{'=' * 60}")
        print(result["reply"])
        print(f"\n--- Token: {result['usage']} ---")

        # 记录本轮对话到历史，传给下一轮
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["reply"]})
        
    