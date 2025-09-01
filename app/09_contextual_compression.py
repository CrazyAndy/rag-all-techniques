from tqdm import tqdm
from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm
from utils.similarity_utils import similar_search
from utils.logger_utils import info

# 0. 构建全局向量模型
embedding_model = EmbeddingModel()


def compress_chunk(chunk_text, query, compression_type="selective"):
    """
    压缩检索到的文本块，仅保留与查询相关的内容。

    Args:
        chunk (str): 要压缩的文本块
        query (str): 用户查询
        compression_type (str): 压缩类型 ("selective", "summary" 或 "extraction")

    Returns:
        str: 压缩后的文本块
    """
    # 为不同的压缩方法定义系统提示
    if compression_type == "selective":
        system_prompt = """您是专业信息过滤专家。
        您的任务是分析文档块并仅提取与用户查询直接相关的句子或段落，移除所有无关内容。

        输出要求：
        1. 仅保留有助于回答查询的文本
        2. 保持相关句子的原始措辞（禁止改写）
        3. 维持文本的原始顺序
        4. 包含所有相关文本（即使存在重复）
        5. 排除任何与查询无关的文本

        请以纯文本格式输出，不添加任何注释。"""

    elif compression_type == "summary":
        system_prompt = """您是专业摘要生成专家。
        您的任务是创建文档块的简洁摘要，且仅聚焦与用户查询相关的信息。

        输出要求：
        1. 保持简明扼要但涵盖所有相关要素
        2. 仅聚焦与查询直接相关的信息
        3. 省略无关细节
        4. 使用中立、客观的陈述语气

        请以纯文本格式输出，不添加任何注释。"""

    else:  # extraction
        system_prompt = """您是精准信息提取专家。
        您的任务是从文档块中精确提取与用户查询相关的完整句子。

        输出要求：
        1. 仅包含原始文本中的直接引用
        2. 严格保持原始文本的措辞（禁止修改）
        3. 仅选择与查询直接相关的完整句子
        4. 不同句子使用换行符分隔
        5. 不添加任何解释性文字

        请以纯文本格式输出，不添加任何注释。"""

    # 定义带有查询和文档块的用户提示
    user_prompt = f"""
        查询: {query}

        文档块:
        {chunk_text}

        请严格提取与本查询相关的核心内容。
    """

    # 使用 OpenAI API 生成响应
    response = query_llm(system_prompt, user_prompt)

    # 从响应中提取压缩后的文本块
    compressed_chunk = response.strip()

    # 计算压缩后占原来文本的比例
    compression_ratio = len(compressed_chunk) / len(chunk_text) * 100

    return compressed_chunk, compression_ratio


def batch_compress_chunks_with_filtering(retrieved_chunks, query, compression_type="selective"):
    """
    最优化版本：直接构建两个列表，避免解包操作
    逐个压缩多个文本块。

    Args:
        chunks (List[str]): 要压缩的文本块列表
        query (str): 用户查询
        compression_type (str): 压缩类型 ("selective", "summary", 或 "extraction")
    """
    compressed_chunks = []
    compression_ratios = []

    for i in tqdm(range(len(retrieved_chunks))):
        chunk_text = retrieved_chunks[i]["text"]
        # 压缩块
        compressed_chunk, compression_ratio = compress_chunk(
            chunk_text, query, compression_type)

        # 直接过滤并添加到对应列表
        if compressed_chunk.strip():
            compressed_chunks.append(compressed_chunk)
            compression_ratios.append(compression_ratio)

    if not compressed_chunks:
        # 如果所有块都被压缩为空，使用原始块
        info("Warning: All chunks were compressed to empty strings. Using original chunks.")
        for i, chunk in enumerate(retrieved_chunks):
            compressed_chunks.append(chunk["text"])
            compression_ratios.append(0.0)

    return compressed_chunks, compression_ratios


if __name__ == "__main__":

    query = "孙悟空的兵器是什么？"

    # 1. 提取文本
    info("---1--->正在提取西游记文本...")
    extract_text = extract_text_from_markdown()
    # print(f"文本长度: {len(extract_text)} 字符")

    # 2. 分割文本
    info("---2--->正在分割文本...")
    knowledge_chunks = chunk_text_by_length(extract_text, 1000, 200)
    # print(f"分割为 {len(text_chunks)} 个文本块")

    # 3. 将知识库文本块向量化
    info("---3--->正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(knowledge_chunks)

    # 4. 构建问题向量
    info("---4--->正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. 向量相似度检索
    info("---5--->向量相似度检索...")
    top_chunks = similar_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, 10)

    info("---5-->搜索结果:")
    for i, result in enumerate(top_chunks):
        info(f" {i+1}. 相似度分数: {result['score']:.4f}")
        info(f"     文档: {result['text'][:100]}...")
        info("")

    compress_chunks, compression_ratios = batch_compress_chunks_with_filtering(
        top_chunks, query, "selective")

    info("\n\n ---6-->压缩后得到的chunk:")
    for i, result in enumerate(compress_chunks):
        info(f" {i+1}. 压缩后占原来长度比例 : {compression_ratios[i]:.2f}%")
        info(f"     文档: {result[:100]}...")
        info("")

    system_prompt = """
    你是一个AI助手，请严格根据以下信息回答问题。如果信息中没有答案，请回答“我不知道”。"""

    user_prompt = "\n".join(
        [f"上下文内容 {i + 1} :\n{result['text']}\n========\n"
         for i, result in enumerate(top_chunks)])

    user_prompt = f"{user_prompt}\n\n Question: {query}"

    # 7. 调用LLM模型，生成回答
    result = query_llm(system_prompt, user_prompt)
    info(f"---7--->final result: {result}")
