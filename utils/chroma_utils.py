import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path


class ChromaVectorDB:
    def __init__(self, collection_name="xiyouji_collection", persist_directory="./chroma_db"):
        """
        初始化 Chroma 向量数据库

        Args:
            collection_name (str): 集合名称
            persist_directory (str): 持久化目录
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 初始化 Chroma 客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 初始化本地嵌入模型
        # 优先使用本地模型路径，如果不存在则使用在线模型
        model_path = os.getenv("LOCAL_MODEL_PATH", None)
        if model_path and os.path.exists(model_path):
            print(f"使用本地模型: {model_path}")
            self.embedding_model = SentenceTransformer(model_path)
        else:
            print("使用在线模型: all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            # 打印模型的实际路径，方便下次使用
            actual_model_path = self.embedding_model.get_sentence_embedding_dimension
            print(f"模型已下载到缓存目录，您可以将以下路径设置为 LOCAL_MODEL_PATH:")
            print(f"LOCAL_MODEL_PATH=\"{self.embedding_model.model_path}\"")

    def add_texts(self, texts, embeddings):
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=[f"doc_{i}" for i in range(len(texts))],
            metadatas=[{"source": "xiyouji.md", "chunk_id": i}
                       for i in range(len(texts))]
        )
        
    def create_collection(self, collection_name):
        """
        创建新的集合
        
        Args:
            collection_name (str): 集合名称
        """
        # 先尝试删除已存在的集合
        try:
            self.client.delete_collection(collection_name)
            print(f"已删除旧集合: {collection_name}")
        except:
            pass
        
        # 创建新集合
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "西游记文本向量数据库"}
        )
        print(f"已创建新集合: {collection_name}")

    def create_embeddings(self, text_chunks):
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(text_chunks).tolist()
        return embeddings

    def search(self, query_embedding, n_results=5):
        """
        搜索相似文档

        Args:
            query (str): 查询文本
            n_results (int): 返回结果数量

        Returns:
            dict: 搜索结果
        """

        # 搜索相似文档
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return results

    def get_collection_info(self):
        """
        获取集合信息

        Returns:
            dict: 集合信息
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }

    def delete_collection(self):
        """
        删除集合
        """
        self.client.delete_collection(self.collection_name)
        print(f"已删除集合: {self.collection_name}")

    def clear_all_collections(self):
        """
        删除所有集合
        """
        try:
            # 获取所有集合
            collections = self.client.list_collections()
            for collection in collections:
                self.client.delete_collection(collection.name)
                print(f"已删除集合: {collection.name}")
        except Exception as e:
            print(f"清除集合时出错: {e}")
