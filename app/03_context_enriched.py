import numpy as np
from utils.common_utils import create_progress_bar
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm
from utils.logger_utils import info
from utils.similarity_utils import cosine_similarity

# 0. 构建全局向量模型
embedding_model = EmbeddingModel()


def chunk_text(text, single_chunk_size, overlap):
    '''
    将文本按单个块大小进行分割，并返回一个包含所有块的列表。

    Args:
        text (str): 要分割的文本
        single_chunk_size (int): 单个块的大小
        overlap (int): 块之间的重叠大小

    Returns:
        list: 包含所有块的列表

    Example:
        >>> chunk_text("Hello, world!", 5, 2)
        ['Hello', 'o, wo', 'rld!']

    '''
    chunks = []
    for i in range(0, len(text), single_chunk_size - overlap):
        chunks.append(text[i:i + single_chunk_size])
    return chunks


def context_enriched_search(knowledge_chunks, knowledge_embeddings, query_embeddings, k=1, context_size=1):
    similarity_scores = []
    
    # 确保查询向量是一维的
    if isinstance(query_embeddings, list):
        query_vector = query_embeddings[0]
    elif hasattr(query_embeddings, 'shape') and len(query_embeddings.shape) > 1:
        query_vector = query_embeddings[0]
    else:
        query_vector = query_embeddings
    
    # 计算查询与每个文本块嵌入之间的相似度分数
    for i, chunk_embedding in enumerate(knowledge_embeddings):
        # 计算查询嵌入与当前文本块嵌入之间的余弦相似度
        similarity_score = cosine_similarity(chunk_embedding, query_vector)
        # 将索引和相似度分数存储为元组
        similarity_scores.append((i, similarity_score))

    # 按相似度分数降序排序（相似度最高排在前面）
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取最相关块的索引
    top_index = similarity_scores[0][0]

    # 定义上下文包含的范围
    # 确保不会超出 text_chunks 的边界
    start = max(0, top_index - context_size)
    end = min(len(knowledge_chunks), top_index + context_size + 1)

    # 返回最相关的块及其相邻的上下文块
    results = []
    for index in range(start, end):
        results.append({
            'text': knowledge_chunks[index],
            'score': similarity_scores[index][1],
            'index': index
        })
    return results


if __name__ == "__main__":

    query = "孙悟空的兵器是从哪里来的？"
    info(f"--0--> Question: {query}")

    # 1. 提取文本
    info("--1--> 正在提取西游记文本...")
    extract_text = extract_text_from_markdown()

    # 2. 分割文本
    info("---2--->正在分割文本...")
    knowledge_chunks = chunk_text(extract_text, 2000, 200) # 这里single_chunk_size要是设置成1000，就无法检索到相关内容

    # 3. 将知识库文本块向量化
    info("--3--> 正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(
        knowledge_chunks, show_progress=True)
    print("")
    # 4. 构建问题向量
    info("--4--> 正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. 向量相似度检索
    info("--5--> 语义相似度检索...")
    top_chunks = context_enriched_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, 5)

    info(f"--5--> 搜索结果:")
    for i, result in enumerate(top_chunks):
        info(
            f"  {i+1}. 相似度分数: {result['score']:.4f} ")
        info(f"    文档: {result['text'][:100]}...")

    system_prompt = """
    你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。"""

    user_prompt = "\n".join(
        [f"上下文内容 {i + 1} :\n{result['text']}\n========\n"
         for i, result in enumerate(top_chunks)])

    user_prompt = f"{user_prompt}\n\n Question: {query}"

    # 7. 调用LLM模型，生成回答
    result = query_llm(system_prompt, user_prompt)
    info(f"--6--> final result: {result}")
