"""
RAG 评估模块

使用 RAGAs 框架评估 RAG 系统的质量
"""

from pathlib import Path
import sys

# 将项目根目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall
)
from langchain_community.chat_models.tongyi import ChatTongyi

from config import LLM_MODEL, DASHSCOPE_API_KEY

def create_llm():
    """
    创建 LLM 实例（用于 RAGAs 评估）
    
    使用 langchain-community 的 ChatDashScope
    调用阿里云百炼模型
    """
    llm = ChatTongyi(
        model_name=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY
    )
    return llm

def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] = None,
    metrics: list = None
) -> dict:
    """
    评估 RAG 系统质量
    
    参数:
        questions: 用户问题列表
        answers: LLM 生成的答案列表
        contexts: 检索到的上下文列表（每个问题的上下文列表）
        ground_truths: 参考答案列表（可选）
        metrics: 评估指标列表
    
    返回:
        评估结果字典
    """

    # 默认使用 3 个核心指标
    if metrics is None:
        metrics = [
            faithfulness, # 答案的正确性
            context_precision, # 上下文的精确性
            context_recall # 上下文的召回率
        ]

    # 创建数据集
    eval_data = {
        "user_input": questions,
        "response": answers,
        "contexts": contexts,
        "reference": ground_truths or [""] * len(questions),
        "ground_truth": ground_truths or [""] * len(questions)
    }

    dataset = Dataset.from_dict(eval_data)

    # 创建 LLM
    llm = create_llm()

    # 评估 RAG 系统质量
    results = evaluate(
        dataset=dataset,
        llm=llm,
        metrics=metrics
    )
    return results

# ===== 测试代码 =====
if __name__ == "__main__":
    # 测试数据（模拟 RAG 系统输出）
    questions = [
        "为什么服务启动失败？",
        "数据库连接有问题吗？",
    ]
    
    answers = [
        "根据日志分析，服务启动失败的原因是数据库连接失败。具体错误为 'Connection refused'，说明数据库服务未启动或端口被拒绝。",
        "是的，数据库连接存在问题。日志显示多次尝试连接数据库均失败，最终导致服务无法正常启动。",
    ]
    
    contexts = [
        [
            "2024-01-01 10:00:00 ERROR 数据库连接失败：Connection refused",
            "2024-01-01 10:00:01 ERROR 服务启动失败：无法连接到数据库"
        ],
        [
            "2024-01-01 10:00:00 ERROR 数据库连接失败：Connection refused",
            "2024-01-01 10:00:05 WARN 重试连接数据库失败"
        ],
    ]
    
    ground_truths = [
        "数据库连接失败导致服务启动失败",
        "数据库连接存在问题",
    ]
    
    print("开始评估...")
    results = evaluate_rag(questions, answers, contexts, ground_truths)
    
    print("\n========== 评估结果 ==========")
    print(results)