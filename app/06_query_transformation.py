from tqdm import tqdm
from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm, query_llm_with_top_chunks
from utils.logger_utils import info
from utils.similarity_utils import cosine_similarity, similar_search
import re

# 0. 构建全局向量模型
embedding_model = EmbeddingModel()




def transformed_search_by_rewrite(original_query, knowledge_chunks, knowledge_embeddings, top_k=3):
    # 定义系统提示，指导AI助手的行为
    system_prompt = "您是一个专注于优化搜索查询的AI助手。您的任务是通过重写用户查询，使其更加具体、详细，并提升检索相关信息的有效性。"

    # 定义用户提示，包含需要重写的原始查询
    user_prompt = f"""
    请优化以下搜索查询，使其满足：
    1. 增强查询的具体性和详细程度
    2. 包含有助于获取准确信息的相关术语和核心概念

    原始查询：{original_query}
    
    直接只给我重写后的查询，不用回答其他文字
    """
    # 1. 重写查询
    rewrited_query = query_llm(system_prompt, user_prompt)
    
    info(f"重写后的查询: {rewrited_query}")

    # 2. 构建查询向量
    query_embeddings = embedding_model.create_embeddings([rewrited_query])

    # 3. 语义相似度检索
    top_chunks = similar_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, top_k)

    # 4. 生成回答
    return query_llm_with_top_chunks(top_chunks, rewrited_query)


def transformed_search_by_step_back(original_query, knowledge_chunks, knowledge_embeddings, top_k=3):
    """
    生成一个更广泛的“回退”查询以检索更宽泛的上下文信息。

    Args:
        original_query (str): 原始用户查询
        model (str): 用于生成回退查询的模型

    Returns:
        str: 回退查询
    """
    # 定义系统提示，以指导AI助手的行为
    system_prompt = "您是一个专注于搜索策略的AI助手。您的任务是将特定查询转化为更宽泛、更通用的版本，以帮助检索相关背景信息。"

    # 定义用户提示，包含要概括的原始查询
    user_prompt = f"""
    请基于以下具体查询生成更通用的版本，要求：
    1. 扩大查询范围以涵盖背景信息
    2. 包含潜在相关领域的关键概念
    3. 保持语义完整性

    原始查询: {original_query}

    直接只给我回退后的查询，不用回答其他文字
    """

    # 1. step back 查询
    step_back_query = query_llm(system_prompt, user_prompt)
    
    info(f"回退后的查询: {step_back_query}")

    # 2. 构建查询向量
    query_embeddings = embedding_model.create_embeddings([step_back_query])

    # 3. 语义相似度检索
    top_chunks = similar_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, top_k)

    # 4. 生成回答
    return query_llm_with_top_chunks(top_chunks, step_back_query)


def transformed_search_by_decompose_query(original_query, knowledge_chunks, knowledge_embeddings, top_k=3, num_subqueries=4):
    """
    将复杂查询分解为更简单的子查询。

    Args:
        original_query (str): 原始的复杂查询
        num_subqueries (int): 要生成的子查询数量
        model (str): 用于查询分解的模型

    Returns:
        List[str]: 更简单子查询的列表
    """
    # 定义系统提示，指导AI助手的行为
    system_prompt = "您是一个专门负责分解复杂问题的AI助手。您的任务是将复杂的查询拆解成更简单的子问题，这些子问题的答案组合起来能够解决原始查询。"

    # 使用需要分解的原始查询定义用户提示
    user_prompt = f"""
    将以下复杂查询分解为{num_subqueries}个更简单的子问题。每个子问题应聚焦原始问题的不同方面。

    原始查询: {original_query}

    请生成{num_subqueries}个子问题，每个问题单独一行，按以下格式：
    1. [第一个子问题]
    2. [第二个子问题]
    依此类推...
    
    直接只给我拆分后的子问题，不用回答其他文字
    """

    # 1. 拆分成子问题 查询
    decompose_query_response = query_llm(system_prompt, user_prompt)

    pattern = r'^\d+\.\s*(.*)'
    children_query = [re.match(pattern, line).group(
        1) for line in decompose_query_response.split('\n') if line.strip()]

    info(f"拆分后的子问题: {children_query}")

    # 2. 为每个子问题构建查询向量
    children_query_embeddings = embedding_model.create_embeddings(
        children_query)

    # 3. 遍历子查询，语义相似度检索
    all_top_chunks = []
    for query_embedding in children_query_embeddings:
        top_chunks = similar_search(
            knowledge_chunks, knowledge_embeddings, query_embeddings, 2)
        all_top_chunks.extend(top_chunks)

    all_top_chunks.sort(key=lambda x: x["score"], reverse=True)

    final_top_chunks = []
    for i, chunk in enumerate(all_top_chunks):
        if i < top_k:
            if any(item["index"] == chunk["index"] for item in final_top_chunks):
                continue
            final_top_chunks.append(chunk)

    # 4. 生成回答
    return query_llm_with_top_chunks(final_top_chunks, original_query)


if __name__ == "__main__":

    query = "孙悟空和牛魔王谁更厉害？"
    info(f"--0--> Question: {query}")

    # 1. 提取文本
    info("--1--> 正在提取西游记文本...")
    extract_text = extract_text_from_markdown()

    # 2. 分割文本
    info("---2--->正在分割文本...")
    # 这里single_chunk_size要是设置成1000，就无法检索到相关内容
    knowledge_chunks = chunk_text_by_length(extract_text, 2000, 200)

    # 3. 将知识库文本块向量化
    info("--3--> 正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(
        knowledge_chunks)

    # 4. 构建问题向量
    info("--4--> 正在构建用户问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. rewrite 查询
    info("\n\n")
    info("--5--> rewrite 查询...")
    rewrite_result = transformed_search_by_rewrite(
        query, knowledge_chunks, knowledge_embeddings)
    info(f"rewrite 查询结果: {rewrite_result}")
    info("\n\n")

    # 6. step back 查询
    info("--6--> step back 查询...")
    step_back_result = transformed_search_by_step_back(
        query, knowledge_chunks, knowledge_embeddings)
    info(f"step back 查询结果: {step_back_result}")

    # 7. decompose 查询
    info("\n\n")
    info("--7--> decompose 查询...")
    decompose_result = transformed_search_by_decompose_query(
        query, knowledge_chunks, knowledge_embeddings)
    info(f"decompose 查询结果: {decompose_result}")
