

import logging
import numpy as np
from utils.common_utils import create_progress_bar
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm
from utils.logger_utils import info
from utils.similarity_utils import cosine_similarity


# 0. 构建全局向量模型
info("--0--> 构建全局向量模型...")
embedding_model = EmbeddingModel()


def compute_breakpoints(similarities, method="percentile", threshold=90):
    # 根据选定的方法确定阈值
    if method == "percentile":
        # 计算相似度分数的第 X 百分位数
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # 计算相似度分数的均值和标准差。
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # 将阈值设置为均值减去 X 倍的标准差
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # 计算第一和第三四分位数（Q1 和 Q3）。
        q1, q3 = np.percentile(similarities, [25, 75])
        # 使用 IQR 规则（四分位距规则）设置阈值
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        # 如果提供了无效的方法，则抛出异常
        raise ValueError(
            "Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # 找出相似度低于阈值的索引
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]


def split_into_chunks(sentences, breakpoints):
    """
    将句子分割为语义块

    Args:
    sentences (List[str]): 句子列表
    breakpoints (List[int]): 进行分块的索引位置

    Returns:
    List[str]: 文本块列表
    """
    chunks = []  # Initialize an empty list to store the chunks
    start = 0  # Initialize the start index

    # 遍历每个断点以创建块
    for bp in breakpoints:
        # 将从起始位置到当前断点的句子块追加到列表中
        chunks.append("。".join(sentences[start:bp + 1]) + "。")
        start = bp + 1  # 将起始索引更新为断点后的下一个句子

    # 将剩余的句子作为最后一个块追加
    chunks.append("。".join(sentences[start:]))
    return chunks  # Return the list of chunks


def chunk_text(extracted_text, method="percentile", threshold=90):
    # 1. 分割文本
    info("--2.1--> 正在分割文本...")
    knowledge_sentences = extracted_text.split("。")
    # 2. 将每一句都分别向量化
    knowledge_embeddings = []
    process_bar = create_progress_bar(len(knowledge_sentences), "--2.2--> 为每一句话都创建向量", 100)
    for index, sentence in enumerate(knowledge_sentences):
        if sentence:    
            single_embedding = embedding_model.create_embeddings(sentence, show_progress=False)
            process_bar.update_by_count(index + 1)
            knowledge_embeddings.append(single_embedding)
    process_bar.finish()
    
    # 3. 计算每一句与下一句的相似度
    similarities = []
    process_bar = create_progress_bar(len(knowledge_embeddings) - 1, "--2.3--> 计算每一句与下一句的相似度", 100)
    for i in range(len(knowledge_embeddings) - 1):
        similarity_score = cosine_similarity(knowledge_embeddings[i], knowledge_embeddings[i + 1])
        process_bar.update_by_count(i)
        similarities.append(similarity_score)
    process_bar.finish()
    
    # 4. 根据相似度计算断点
    info("--2.4--> 根据相似度计算断点...")
    breakpoints = compute_breakpoints(
        similarities, method=method, threshold=threshold)

    # 5. 根据断点分割文本
    info("--2.5--> 根据断点分割文本...")
    knowledge_chunks = split_into_chunks(knowledge_sentences, breakpoints)
    return knowledge_chunks


def semantic_search(knowledge_chunks, knowledge_embeddings, query_embeddings, k=5):
    """
    语义搜索，计算相似度并返回最相关的文本块

    Args:
        text_chunks: 知识库文本块列表
        knowledge_base_embeddings: 知识库嵌入向量列表
        query_embeddings: 查询嵌入向量
        k: 返回结果数量

    Returns:
        list: 包含文本块和相似度分数的字典列表
    """
    similarity_scores = []

    # 确保查询向量是一维的
    if isinstance(query_embeddings, list):
        query_vector = query_embeddings[0]
    elif hasattr(query_embeddings, 'shape') and len(query_embeddings.shape) > 1:
        query_vector = query_embeddings[0]
    else:
        query_vector = query_embeddings

    for i, chunk_embedding in enumerate(knowledge_embeddings):
        similarity_score = cosine_similarity(chunk_embedding, query_vector)
        similarity_scores.append((i, similarity_score))

    # 按相似度降序排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 返回包含文本块和相似度分数的结果
    results = []
    for index, score in similarity_scores[:k]:
        results.append({
            'text': knowledge_chunks[index],
            'score': score,
            'index': index
        })
    return results


if __name__ == "__main__":

    query = "孙悟空被如来佛祖压在了哪里？"

    # 1. 提取文本
    info("--1--> 正在提取西游记文本...")
    extract_text = extract_text_from_markdown()

    # 2. 分割文本
    info("--2--> 正在分割文本...")
    knowledge_chunks = chunk_text(extract_text, "percentile", 90)

    # 3. 将知识库文本块向量化
    info("--3--> 正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(knowledge_chunks,show_progress=True)

    # 4. 构建问题向量
    info("--4--> 正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. 向量相似度检索
    info("--5--> 语义相似度检索...")
    top_chunks = semantic_search(
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
