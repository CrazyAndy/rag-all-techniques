from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm


class EmbeddingModel:
    def __init__(self):
        # 初始化本地嵌入模型
        # 优先使用本地模型路径，如果不存在则使用在线模型
        model_name = os.getenv('EMBEDDING_MODEL', None)
        model_path = os.getenv("LOCAL_MODEL_PATH", None)

        if model_path and os.path.exists(model_path):
            # print(f"使用本地模型: {model_path}")
            self.model = SentenceTransformer(model_path)
        else:
            # print(f"使用在线模型: {model_name}")
            self.model = SentenceTransformer(model_name)
            # print(f"模型名称: {self.model.transformers_model.config.name_or_path}")

    def create_embeddings(self, text_chunks, batch_size=16):
        """
        分批处理文本块创建嵌入向量，避免GPU内存不足

        Args:
            text_chunks: 单个文本字符串或文本块列表
            batch_size: 批处理大小，默认16

        Returns:
            单个文本时返回单个嵌入向量，文本列表时返回嵌入向量列表
        """
        # 处理单个文本的情况
        if isinstance(text_chunks, str):
            # 将单个文本包装成列表处理
            embeddings = self._batch_encode(
                [text_chunks], batch_size)
            return embeddings[0]  # 返回单个嵌入向量
        # 处理文本列表的情况
        elif isinstance(text_chunks, list):
            return self._batch_encode(text_chunks, batch_size)
        else:
            raise ValueError("text_chunks 必须是字符串或字符串列表")

    def _batch_encode(self, text_chunks, batch_size):
        """
        内部方法：分批编码文本块

        Args:
            text_chunks: 文本块列表
            batch_size: 批处理大小

        Returns:
            list: 所有文本块的嵌入向量列表
        """
        all_embeddings = []

        # 分批处理文本块
        # 不使用进度条
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            # 处理当前批次
            batch_embeddings = self.model.encode(batch).tolist()
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
