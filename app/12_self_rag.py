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


def determine_if_retrieval_needed(query):
    """
    判断给定查询是否需要检索。

    Args:
        query (str): 用户查询

    Returns:
        bool: 如果需要检索，返回True；否则返回False
    """
    # 系统提示，指导AI如何判断是否需要检索
    system_prompt = """你是一个判断查询是否需要检索的AI助手。
    针对事实性问题、具体信息请求或关于事件、人物、概念的查询，回答"Yes"。
    对于观点类、假设性场景或常识性简单查询，回答"No"。
    仅回答"Yes"或"No"。"""

    # 包含查询的用户提示
    user_prompt = f"查询: {query}\n\n准确回答此查询是否需要检索？"

    # 使用模型生成响应
    response = query_llm(system_prompt, user_prompt, temperature=0)

    # 从模型响应中提取答案并转换为小写
    answer = response.strip().lower()

    # 如果答案包含“yes”，返回True；否则返回False
    return "yes" in answer


def evaluate_relevance(query, context):
    """
    评估上下文与查询的相关性。

    Args:
        query (str): 用户查询
        context (str): 上下文文本

    Returns:
        str: 'relevant'（相关）或 'irrelevant'（不相关）
    """
    # 系统提示，指导AI如何判断文档是否与查询相关
    system_prompt = """你是一个AI助手，任务是判断文档是否与查询相关。
    判断文档中是否包含有助于回答查询的信息。
    仅回答“Relevant”或“Irrelevant”。"""

    # 如果上下文过长以避免超出标记限制，则截断上下文
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # 包含查询和文档内容的用户提示
    user_prompt = f"""查询: {query}
    文档内容:
    {context}

    该文档与查询相关？仅回答“Relevant”或“Irrelevant”。
    """

    # 使用模型生成响应
    response = query_llm(system_prompt, user_prompt, temperature=0)

    # 从模型响应中提取答案并转换为小写
    answer = response.strip().lower()

    return answer  # 返回相关性评估结果


def assess_support(response, context):
    """
    评估响应在多大程度上得到上下文的支持。

    Args:
        response (str): 生成的响应
        context (str): 上下文文本

    Returns:
        str: 'fully supported'（完全支持）、'partially supported'（部分支持）或 'no support'（无支持）
    """
    # 系统提示，指导AI如何评估支持情况
    system_prompt = """你是一个AI助手，任务是判断回答是否基于给定的上下文。
    评估响应中的事实、主张和信息是否由上下文支持。
    仅回答以下三个选项之一：
    - "Fully supported"（完全支持）：回答所有信息均可从上下文直接得出。
    - "Partially supported"（部分支持）：回答中的部分信息由上下文支持，但部分不是。
    - "No support"（无支持）：回答中包含大量未在上下文中找到、提及或与上下文矛盾的信息。
    """

    # 如果上下文过长以避免超出标记限制，则截断上下文
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # 包含上下文和要评估的响应的用户提示
    user_prompt = f"""上下文:
    {context}

    回答:
    {response}

    该回答与上下文的支持程度如何？仅回答 "Fully supported"、"Partially supported"或 "No support"。
    """

    # 使用模型生成响应
    response = query_llm(system_prompt, user_prompt, temperature=0)

    # 从模型响应中提取答案并转换为小写
    answer = response.strip().lower()

    return answer  # 返回支持评估结果


def rate_utility(query, response):
    """
    评估响应对查询的实用性。

    Args:
        query (str): 用户查询
        response (str): 生成的响应

    Returns:
        int: 实用性评分，范围为1到5
    """
    # 系统提示，指导AI如何评估响应的实用性
    system_prompt = """你是一个AI助手，任务是评估一个回答对查询的实用性。
    从回答准确性、完整性、正确性和帮助性进行综合评分。
    使用1-5级评分标准：
    - 1：毫无用处
    - 2：稍微有用
    - 3：中等有用
    - 4：非常有用
    - 5：极其有用
    仅回答一个从1到5的单个数字，不要过多解释。"""

    # 包含查询和要评分的响应的用户提示
    user_prompt = f"""查询: {query}
    回答:
    {response}

    请用1到5分的评分评估该回答的效用，仅用一个1-5的数字评分。"""

    # 使用OpenAI客户端生成实用性评分
    response = query_llm(system_prompt, user_prompt, temperature=0)

    # 从模型响应中提取评分
    rating = response.strip()

    # 提取评分中的数字
    rating_match = re.search(r'[1-5]', rating)
    if rating_match:
        return int(rating_match.group())  # 返回提取的评分作为整数

    return 3  # 如果解析失败，默认返回中间评分


if __name__ == "__main__":
    query = "孙悟空会多少种变化的法术？"
    info(f"--0--> Question: {query}")

    retrieval_needed = determine_if_retrieval_needed(query)
    info(f"--5--> Retrieval needed: {retrieval_needed}")

    best_response = None  # 最佳响应初始化为None
    best_score = -1  # 最佳分数初始化为-1

    if retrieval_needed:
        # 针对事实性问题、具体信息请求或关于事件、人物、概念的查询
        # 需要检索知识
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
        knowledge_embeddings = embedding_model.create_embeddings(
            knowledge_chunks)

        # 4. 构建问题向量
        info("---4--->正在构建问题向量...")
        query_embeddings = embedding_model.create_embeddings([query])

        # 5. 向量相似度检索
        info("---5---> 向量相似度检索...")
        top_chunks = similar_search(
            knowledge_chunks, knowledge_embeddings, query_embeddings, 5)

        info(f"--5--> 搜索结果: ")
        for i, result in enumerate(top_chunks):
            info(f"   {i+1}. 相似度分数: {result['score']:.4f}")
            info(f"   文档: {result['text'][:100]}...")
            info("")

        info("第6步：评估文档相关性...")
        relevant_contexts = []
        for i, chunk in enumerate(top_chunks):
            relevance = evaluate_relevance(query, chunk['text'])
            if relevance:
                info(f"6.1：LLM认为该chunk和查询相关 {i}")
                info(f"6.2：上下文: {chunk['text'][:100]}...")
                info("")
                relevant_contexts.append(chunk)

        if relevant_contexts:
            info(f"第7步：开始遍历这些相关性文档，共{len(relevant_contexts)}个")            
            for i, chunk in enumerate(relevant_contexts):
                info(f"7.0：开始遍历---relevant chunk-----------< {i} >-----------------")
                info(f"7.0：上下文: {chunk['text'][:100]}...")
                
                response = query_llm_with_top_chunks([chunk], query)
                info(f"7.1：LLM答案: {response}")
                
                # 评估响应对上下文的支持程度
                support_rating = assess_support(
                    response, chunk['text'])  # 评估支持程度
                support_score = {
                    "fully supported": 3,  # 完全支持得分为3
                    "partially supported": 1,  # 部分支持得分为1
                    "no support": 0  # 无支持得分为0
                }.get(support_rating, 0)
                info(f"7.2：评估响应得到上下文的支持程度为 {support_rating},评分为：{support_score}")
                
                # 评估响应的实用性
                utility_rating = rate_utility(query, response)  # 评估实用性
                info(f"7.3：评估响应对查询的实用性评分: {utility_rating}")
                
                # 计算总体评分（支持和实用性越高，评分越高）
                overall_score = support_score * 5 + utility_rating  # 计算总体评分
                info(f"7.4：总体评分: {overall_score}")
                print("\n")

                # 跟踪最佳响应
                if overall_score > best_score:  # 如果当前评分高于最佳评分，则更新最佳响应和评分
                    best_response = response
                    best_score = overall_score
                    
        if not relevant_contexts or best_score <=0:
             info("第8步：未找到合适的上下文或响应评分较差，直接生成响应而不进行检索...")
             best_response = query_llm("你是一个全知全能的神，请你根据自己的理解回答问题", query)  # 不使用检索直接生成响应                        
    else:
        # 观点类、假设性场景或常识性简单查询
        # 不需要，直接回答
        info("第9步：不需要检索，直接回答...")
        best_response = query_llm("你是一个全知全能的神，请你根据自己的理解回答问题", query)

    info(f"--final--> 是否使用了检索: {retrieval_needed}")
    info(f"--final--> Best response: {best_response}")
    info(f"--final--> Best score: {best_score}")
