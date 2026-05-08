from pathlib import Path
import sys

# 将项目根目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chromadb
from chromadb.config import Settings
import dashscope
from dashscope import TextEmbedding

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    DASHSCOPE_API_KEY,
    TOP_K
)

# 设置阿里云百炼 API Key
dashscope.api_key = DASHSCOPE_API_KEY

def get_embedding(text: str) -> list[float]:
    """
    使用阿里云百炼生成文本向量
    
    参数:
        text: 要向量化的文本
    
    返回:
        向量列表
    """
    response = TextEmbedding.call(
        model=EMBEDDING_MODEL,
        input=text
    )
    
    # 提取向量
    embedding = response.output['embeddings'][0]['embedding']
    return embedding

class ChromaDBRetriever:
    """
    ChromaDB 向量检索器
    
    功能：
    1. 使用阿里云百炼 embedding 生成向量
    2. 存储日志向量到 ChromaDB
    3. 根据查询检索相似日志
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        """
        初始化 ChromaDB
        
        参数:
            persist_dir: 数据持久化目录，默认从 config 读取
            collection_name: 集合名称，默认从 config 读取
        """

        # 使用默认配置
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME

        # 创建持久化客户端
        # persist_dir 指定数据保存路径，重启后数据还在
        self.client = chromadb.PersistentClient(path=self.persist_dir) 
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "运维日志向量存储"}
        )

        print(f"ChromaDB 初始化完成，集合: {self.collection_name}")
        
    def add_documents(self, chunks: list[dict]) -> None:
        """
        添加文档到向量数据库
        
        参数:
            chunks: 分块列表，每个元素包含 text、metadata
        """
        if not chunks:
            print("没有文档需要添加")
            return

        # 准备数据
        ids = []          # 文档 ID
        documents = []    # 文档文本
        embeddings = []
        metadatas = []    # 元数据

        for chunk in chunks:
            ids.append(chunk["id"])
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

            embedding = get_embedding(chunk["text"])
            embeddings.append(embedding)
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"成功添加 {len(chunks)} 个文档到 ChromaDB")

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        检索与查询相似的文档
        
        参数:
            query: 用户问题
            top_k: 返回前 K 个最相似的文档
        
        返回:
            检索结果列表，每个元素包含 document、metadata、distance
        """
        top_k = top_k or TOP_K

        query_embedding = get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # 整理结果格式
        retrieved_docs = []

        # ChromaDB 返回的批量数据，取第一个
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for i, doc_id in enumerate(ids):
            retrieved_docs.append({
                "id": doc_id,
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i]
            })

        return retrieved_docs


# ===== 测试代码 =====
if __name__ == "__main__":
    retriever = ChromaDBRetriever()
    
    test_chunks = [
        {"id": "char_0", "text": "数据库连接失败，请检查网络", "metadata": {"time": "2024-01-01 10:00:00", "level": "ERROR"}},
        {"id": "char_1", "text": "服务启动成功", "metadata": {"time": "2024-01-01 10:01:00", "level": "INFO"}},
        {"id": "char_2", "text": "内存使用率超过80%", "metadata": {"time": "2024-01-01 10:02:00", "level": "WARNING"}},
    ]
    retriever.add_documents(test_chunks)

    # 测试检索
    results = retriever.search("数据库出问题了", top_k=3)
    
    print("\n检索结果:")
    for result in results:
        print(f"\n结果 {result['id']}:")
        print(f"  内容: {result['document']}")
        print(f"  距离: {result['distance']:.4f}")
        print(f"  级别: {result['metadata']['level']}")

    count = retriever.collection.count()
    print(f"共 {count} 条数据")

    # 获取所有 ID
    all_data = retriever.collection.get()
    all_ids = all_data["ids"]
    # 用 ID 删除
    if all_ids:
        retriever.collection.delete(ids=all_ids)
    print("所有数据已删除")