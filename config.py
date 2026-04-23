import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent
# data 目录
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# 自动创建目录（如果不存在）
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# 阿里云百炼配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

# 模型配置
LLM_MODEL = "qwen-plus"  # 对话模型
EMBEDDING_MODEL = "text-embedding-v3"  # 向量模型
RERANK_MODEL = "gte-rerank-v1"  # 重排模型

# ChromaDB 配置
CHROMA_PERSIST_DIR = str(PROCESSED_DATA_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "logs"

# 检索配置
TOP_K = 5  # 检索返回前 5 条
RERANK_TOP_K = 3  # 重排后返回前 3 条

# 分块配置
CHUNK_SIZE = 500  # 每块 500 字符
CHUNK_OVERLAP = 50  # 块之间重叠 50 字符