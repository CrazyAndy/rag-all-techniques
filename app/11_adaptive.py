import re
from typing import List
from tqdm import tqdm
from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm, query_llm_with_top_chunks
from utils.similarity_utils import cosine_similarity, similar_search
from utils.logger_utils import info
import json


# 0. 构建全局向量模型
embedding_model = EmbeddingModel()


def classify_query(query):
    """
    将查询分类为四个类别之一：事实性（Factual）、分析性（Analytical）、意见性（Opinion）或上下文相关性（Contextual）。

    Returns:
        str: 查询类别
    """
    # 定义系统提示以指导AI进行分类
    system_prompt = """您是专业的查询分类专家。
        请将给定查询严格分类至以下四类中的唯一一项：
        - Factual：需要具体、可验证信息的查询
        - Analytical：需要综合分析或深入解释的查询
        - Opinion：涉及主观问题或寻求多元观点的查询
        - Contextual：依赖用户具体情境的查询

        请仅返回分类名称，不要添加任何解释或额外文本。
    """

    # 创建包含要分类查询的用户提示
    user_prompt = f"对以下查询进行分类: {query}"

    # 从AI模型生成分类响应
    response = query_llm(system_prompt, user_prompt)

    # 从响应中提取并去除多余的空白字符以获取类别
    category = response.strip()

    # 定义有效的类别列表
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]

    # 确保返回的类别是有效的
    for valid in valid_categories:
        if valid in category:
            return valid

    # 如果分类失败，默认返回“Factual”（事实性）
    return "Factual"


def factual_retrieval_strategy(query, knowledge_chunks, knowledge_embeddings, k=4):
    """
    针对事实性查询的检索策略，专注于精确度。

    Returns:
        List[Dict]: 检索到的文档列表
    """
    print(f"执行事实性检索策略: '{query}'")

    # 使用LLM增强查询以提高精确度
    system_prompt = """您是搜索查询优化专家。
        您的任务是重构给定的事实性查询，使其更精确具体以提升信息检索效果。
        重点关注关键实体及其关联关系。

        请仅提供优化后的查询，不要包含任何解释。
    """

    user_prompt = f"请优化此事实性查询: {query}"

    # 使用LLM生成增强后的查询
    response = query_llm(system_prompt, user_prompt, temperature=0)

    # 提取并打印增强后的查询
    enhanced_query = response.strip()
    print(f"优化后的查询: {enhanced_query}")

    # 为增强后的查询创建嵌入向量
    query_embedding = embedding_model.create_embeddings([enhanced_query])

    # 执行初始相似性搜索以检索文档
    initial_results = similar_search(
         knowledge_chunks, knowledge_embeddings,query_embedding, k=k*2)

    # 返回前k个结果
    return initial_results[:k]


def analytical_retrieval_strategy(query, query_embeddings, knowledge_chunks, knowledge_embeddings,  k=4):
    """
    针对分析性查询的检索策略，专注于全面覆盖。

    Returns:
        List[Dict]: 检索到的文档列表
    """
    print(f"执行分析性检索策略: '{query}'")

    # 定义系统提示以指导AI生成子问题
    system_prompt = """您是复杂问题拆解专家。
    请针对给定的分析性查询生成探索不同维度的子问题。
    这些子问题应覆盖主题的广度并帮助获取全面信息。

    请严格生成恰好3个子问题，每个问题单独一行。
    """

    # 创建包含主查询的用户提示
    user_prompt = f"请为此分析性查询生成子问题：{query}"

    # 使用LLM生成子问题
    response = query_llm(system_prompt, user_prompt, temperature=0.3)

    # 提取并清理子问题
    sub_queries = response.strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    print(f"生成的子问题: {sub_queries}")

    # 为每个子问题检索文档
    all_results = []
    for sub_query in sub_queries:
        # 为子问题创建嵌入向量
        sub_query_embedding = embedding_model.create_embeddings([sub_query])
        # 执行相似性搜索以获取子问题的结果
        results = similar_search(
            knowledge_chunks, knowledge_embeddings,sub_query_embedding, k=2)
        all_results.extend(results)

    # 确保多样性，从不同的子问题结果中选择
    # 移除重复项（相同的文本内容）
    unique_texts = set()
    diverse_results = []

    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)

    # 如果需要更多结果以达到k，则从初始结果中添加更多
    if len(diverse_results) < k:
        # 对主查询直接检索
        main_results = similar_search(
            knowledge_chunks, knowledge_embeddings,query_embeddings,  k=2)

        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)

    # 返回前k个多样化的结果
    return diverse_results[:k]


def opinion_retrieval_strategy(query, knowledge_chunks, knowledge_embeddings, k=4):
    """
    针对观点查询的检索策略，专注于多样化的观点。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储库
        k (int): 返回的文档数量

    Returns:
        List[Dict]: 检索到的文档列表
    """
    print(f"执行观点检索策略: '{query}'")

    # 定义系统提示以指导AI识别不同观点
    system_prompt = """您是主题多视角分析专家。
        针对给定的观点类或意见类查询，请识别人们可能持有的不同立场或观点。

        请严格返回恰好3个不同观点角度，每个角度单独一行。
    """

    # 创建包含主查询的用户提示
    user_prompt = f"请识别以下主题的不同观点：{query}"

    # 使用LLM生成不同的观点
    response = query_llm(system_prompt, user_prompt, temperature=0.3)

    # 提取并清理观点
    viewpoints = response.strip().split('\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    print(f"已识别的观点: {viewpoints}")

    # 检索代表每个观点的文档
    all_results = []
    for viewpoint in viewpoints:
        # 将主查询与观点结合
        combined_query = f"{query} {viewpoint}"
        # 为组合查询创建嵌入向量
        viewpoint_embedding = embedding_model.create_embeddings([combined_query])
        # 执行相似性搜索以获取组合查询的结果
        results = similar_search(knowledge_chunks, knowledge_embeddings,
                                 viewpoint_embedding,  k=2)

        # 标记结果所代表的观点
        for result in results:
            result["viewpoint"] = viewpoint

        # 将结果添加到所有结果列表中
        all_results.extend(results)

    # 选择多样化的意见范围
    # 尽量确保从每个观点中至少获得一个文档
    selected_results = []
    for viewpoint in viewpoints:
        # 按观点过滤文档
        viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])

    # 用最高相似度的文档填充剩余的槽位
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        # 按相似度排序剩余文档
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["score"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])

    # 返回前k个结果
    return selected_results[:k]


def contextual_retrieval_strategy(query, knowledge_chunks, knowledge_embeddings, k=4):
    """
    针对上下文查询的检索策略，结合用户提供的上下文信息。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储库
        k (int): 返回的文档数量
        user_context (str): 额外的用户上下文信息

    Returns:
        List[Dict]: 检索到的文档列表
    """
    print(f"执行上下文检索策略: '{query}'")

    # 如果未提供用户上下文，则尝试从查询中推断上下文
    system_prompt = """您是理解查询隐含上下文的专家。
        对于给定的查询，请推断可能相关或隐含但未明确说明的上下文信息。
        重点关注有助于回答该查询的背景信息。

        请简要描述推断的隐含上下文。
        """

    user_prompt = f"推断此查询中的隐含背景(上下文)：{query}"

    # 使用LLM生成推断出的上下文
    response = query_llm(system_prompt, user_prompt, temperature=0.1)

    # 提取并打印推断出的上下文
    user_context = response.strip()
    print(f"推断出的上下文: {user_context}")
        

    # 重新表述查询以结合上下文
    system_prompt = """您是上下文整合式查询重构专家。
    根据提供的查询和上下文信息，请重新构建更具体的查询以整合上下文，从而获取更相关的信息。

    请仅返回重新构建的查询，不要包含任何解释。
    """

    user_prompt = f"""
    原始查询：{query}
    关联上下文：{user_context}

    请结合此上下文重新构建查询：
    """

    # 使用LLM生成结合上下文的查询
    response = query_llm(system_prompt, user_prompt, temperature=0)

    # 提取并打印结合上下文的查询
    contextualized_query = response.strip()
    print(f"结合上下文的查询: {contextualized_query}")

    # 基于结合上下文的查询检索文档
    query_embedding = embedding_model.create_embeddings([contextualized_query])
    initial_results = similar_search(knowledge_chunks, knowledge_embeddings,
                                     query_embedding,k=k)

    # 按上下文相关性排序，并返回前k个结果
    initial_results.sort(key=lambda x: x["score"], reverse=True)
    return initial_results[:k]



if __name__ == "__main__":

    query = "孙悟空会多少种变化的法术？" # Factual：需要具体、可验证信息的查询
    query = "为什么镇元子和孙悟空结拜，有什么深层含义？" # Analytical：需要综合分析或深入解释的查询
    query = "孙悟空和猪八戒谁更厉害？" # Opinion：涉及主观问题或寻求多元观点的查询
    query = "唐僧对被妖怪抓住，猪八戒什么态度？" # Contextual：依赖用户具体情境的查询

    info(f"--0--> Question: {query}")
    # 1. 提取文本
    info("---1--->正在提取西游记文本...")
    extract_text = extract_text_from_markdown()
    # print(f"文本长度: {len(extract_text)} 字符")

    # 2. 分割文本
    info("---2--->正在分割文本...")
    knowledge_chunks = chunk_text_by_length(extract_text, 2000, 200)
    # print(f"分割为 {len(text_chunks)} 个文本块")

    # 3. 将知识库文本块向量化
    info("---3--->正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(knowledge_chunks)

    # 4. 构建问题向量
    info("---4--->正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 对查询进行分类以确定其类型
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")  # 打印查询被分类为的类型

    k = 24
    # 根据查询类型选择并执行适当的检索策略
    if query_type == "Factual":
        # 使用事实检索策略获取精确信息
        results = factual_retrieval_strategy(
            query, knowledge_chunks, knowledge_embeddings, k=2)
        system_prompt = """您是基于事实信息应答的AI助手。
                请严格根据提供的上下文回答问题，确保信息准确无误。
                若上下文缺乏必要信息，请明确指出信息局限。"""
    elif query_type == "Analytical":
        # 使用分析检索策略实现全面覆盖
        results = analytical_retrieval_strategy(
            query, query_embeddings, knowledge_chunks, knowledge_embeddings, k)
        system_prompt = """您是专业分析型AI助手。
                请基于提供的上下文，对主题进行多维度深度解析：
                - 涵盖不同层面的关键要素（不同方面和视角）
                - 整合多方观点形成系统分析
                若上下文存在信息缺口或空白，请在分析时明确指出信息短缺。"""
    elif query_type == "Opinion":
        # 使用观点检索策略获取多样化的观点
        results = opinion_retrieval_strategy(query, knowledge_chunks, knowledge_embeddings, k)
        system_prompt = """您是观点整合型AI助手。
                请基于提供的上下文，结合以下标准给出不同观点：
                - 全面呈现不同立场观点
                - 保持各观点表述的中立平衡，避免出现偏见
                - 当上下文视角有限时，直接说明"""
    elif query_type == "Contextual":
        # 使用上下文检索策略，并结合用户上下文
        results = contextual_retrieval_strategy(
            query, knowledge_chunks, knowledge_embeddings, k)
        system_prompt = """您是情境上下文感知型AI助手。
                请结合查询背景与上下文信息：
                - 建立问题情境与文档内容的关联
                - 当上下文无法完全匹配具体情境时，请明确说明适配性限制"""
    else:
        # 如果分类失败，默认使用事实检索策略
        results = factual_retrieval_strategy(query, knowledge_chunks,knowledge_embeddings, k)
        system_prompt = """您是通用型AI助手。请基于上下文回答问题，若信息不足请明确说明。"""

    info(f"--8--> 重新构建的文本:")
    for i, result in enumerate(results):
        info(
            f"  {i+1}. 相似度分数: {result['score']:.4f} ")
        info(f"    文档: {result['text'][:200]}...\n")

    # 通过结合上下文和查询创建用户提示
    # 从检索到的文档中准备上下文，通过连接它们的文本并使用分隔符
    context = "\n\n---\n\n".join([r["text"] for r in results])
    user_prompt = f"""
                上下文:
                {context}

                问题: {query}

                请基于上下文提供专业可靠的回答。
    """
    # 9. 根据最佳的段落，结合问题给大模型进行回答
    info("--9--> 根据最佳的段落，结合问题给大模型进行回答...\n")
    answer = query_llm(system_prompt, user_prompt)
    info(f"大模型最终答案: {answer}")
