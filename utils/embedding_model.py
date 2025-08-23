import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path


class EmbeddingModel:
    def __init__(self):
        # 初始化本地嵌入模型
        # 优先使用本地模型路径，如果不存在则使用在线模型
        model_path = os.getenv("LOCAL_MODEL_PATH", None)
        if model_path and os.path.exists(model_path):
            print(f"使用本地模型: {model_path}")
            self.model = SentenceTransformer(model_path)
        else:
            print("使用在线模型: all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_embeddings(self, text_chunks):
        return self.model.encode(text_chunks).tolist()