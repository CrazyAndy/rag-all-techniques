from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from dotenv import load_dotenv
from utils.llm_utils import query_llm
from utils.similarity_utils import similar_search
from utils.logger_utils import info
import re
import os
from tavily import TavilyClient

load_dotenv()

embedding_model = EmbeddingModel()

# 初始化Tavily客户端
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_KEY")
)


def evaluate_document_relevance(query, document):
    """
    评估文档与查询的相关性。

    Args:
        query (str): 用户查询
        document (str): 文档文本

    Returns:
        float: 相关性评分（0 到 1）
    """
    # 定义系统提示语，指导模型如何评估相关性
    system_prompt = """
    你是一位评估文档相关性的专家。
    请在 0 到 1 的范围内对给定文档与查询的相关性进行评分。
    0 表示完全不相关，1 表示完全相关。
    仅返回一个介于 0 和 1 之间的浮点数评分，不要过多解释与生成。
    """

    # 构造用户提示语，包含查询和文档内容
    user_prompt = f"查询：{query}\n\n文档：{document}"

    try:
        # 调用 OpenAI API 进行相关性评分
        response = query_llm(system_prompt, user_prompt, temperature=0)

        # 提取评分结果
        score_text = response.strip()
        # 使用正则表达式提取响应中的浮点数值
        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
        if score_match:
            return float(score_match.group(1))  # 返回提取到的浮点型评分
        return 0.5  # 如果解析失败，默认返回中间值

    except Exception as e:
        # 捕获异常并打印错误信息，出错时返回默认值
        print(f"评估文档相关性时出错：{e}")
        return 0.5  # 出错时默认返回中等评分


def rewrite_search_query(query):
    """
    将查询重写为更适合网络搜索的形式。

    Args:
        query (str): 原始查询语句

    Returns:
        str: 重写后的查询语句
    """
    # 定义系统提示，指导模型如何重写查询
    system_prompt = """
    你是一位编写高效搜索查询的专家。
    请将给定的查询重写为更适合搜索引擎的形式。
    重点使用关键词和事实，去除不必要的词语，使查询更简洁明确。
    """

    user_prompt = f"原始查询：{query}\n\n重写后的查询："

    try:
        # 调用 OpenAI API 来重写查询
        response = query_llm(system_prompt, user_prompt, max_tokens=50)

        # 返回重写后的查询结果（去除首尾空白）
        return response.strip()

    except Exception as e:
        # 如果发生错误，打印错误信息并返回原始查询
        print(f"重写搜索查询时出错：{e}")
        return query  # 出错时返回原始查询


def tavily_search(query, num_results=3):
    """
    使用 Tavily 执行网络搜索。

    Args:
        query (str): 搜索查询语句
        num_results (int): 要返回的结果数量

    Returns:
        str: 合并后的搜索结果文本
    """
    try:
        # 使用Tavily执行搜索
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=num_results
        )

        # 初始化变量用于存储搜索结果和来源信息
        results_text = ""

        # 处理搜索结果
        for result in response.get("results", []):
            # 添加内容到结果文本
            if result.get("content"):
                results_text += f"{result['content']}\n\n"

        return results_text

    except Exception as e:
        # 如果搜索失败，打印错误信息并返回空结果
        print(f"执行Tavily搜索时出错：{e}")
        return "无法获取搜索结果。", []


def refine_knowledge(knowledge_text):
    """
    使用LLM精炼和优化知识内容。

    Args:
        knowledge_text (str): 原始知识文本

    Returns:
        str: 精炼后的知识文本
    """
    system_prompt = """
    你是一位知识精炼专家。请将给定的知识内容进行精炼和优化：
    1. 去除冗余信息
    2. 突出关键要点
    3. 保持信息的准确性和完整性
    4. 使内容更加清晰易懂
    """

    user_prompt = f"请精炼以下知识内容：\n\n{knowledge_text}"

    try:
        refined_text = query_llm(system_prompt, user_prompt, temperature=0.1)
        return refined_text
    except Exception as e:
        print(f"精炼知识时出错：{e}")
        return knowledge_text  # 出错时返回原始文本


def perform_web_search(query):
    """
    使用重写后的查询执行网络搜索。

    Args:
        query (str): 用户原始查询语句

    Returns:
        Tuple[str, List[Dict]]: 搜索结果文本 和 来源元数据列表
    """
    # 重写查询以提升搜索效果
    rewritten_query = rewrite_search_query(query)
    print(f"重写后的搜索查询：{rewritten_query}")

    # 使用重写后的查询执行网络搜索
    results_text = tavily_search(rewritten_query)

    # 返回搜索结果
    return results_text


def get_final_knowledge(query, top_chunks, max_score, best_doc_idx):
    # 记录来源用于引用
    final_knowledge = ""

    # 步骤 4: 根据情况执行相应的知识获取策略
    if max_score > 0.7:
        # 情况 1: 高相关性 - 直接使用文档内容
        best_doc = top_chunks[best_doc_idx]["text"]
        final_knowledge = best_doc
        info(f"高相关性 ({max_score:.2f}) - 直接使用文档内容:{final_knowledge}")
    elif max_score < 0.3:
        # 情况 2: 低相关性 - 使用网络搜索
        web_results = perform_web_search(query)
        final_knowledge = refine_knowledge(web_results)
        info(f"低相关性 ({max_score:.2f}) - 进行网络搜索:{final_knowledge}")
    else:
        # 情况 3: 中等相关性 - 结合文档与网络搜索结果
        best_doc = top_chunks[best_doc_idx]["text"]
        refined_doc = refine_knowledge(best_doc)

        # 获取网络搜索结果
        web_results = perform_web_search(query)
        refined_web = refine_knowledge(web_results)

        # 合并知识
        final_knowledge = f"来自文档的内容:\n{refined_doc}\n\n来自网络搜索的内容:\n{refined_web}"
        info(f"中等相关性 ({max_score:.2f}) - 结合文档与网络搜索:{final_knowledge}")
    
    return final_knowledge
    

if __name__ == "__main__":

    query = "猪八戒原来是干什么的？"

    # 1. 提取文本
    info("---1--->正在提取西游记文本...")
    extract_text = extract_text_from_markdown()

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
        knowledge_chunks, knowledge_embeddings, query_embeddings, 5)

    # 6: 评估文档相关性
    print("正在评估文档的相关性...")
    relevance_scores = []
    for doc in top_chunks:
        score = evaluate_document_relevance(query, doc["text"])
        relevance_scores.append(score)
        doc["relevance"] = score
        print(f"文档得分为 {score:.2f} 的相关性")

    # 7: 根据最高相关性得分确定操作策略
    max_score = max(relevance_scores) if relevance_scores else 0
    best_doc_idx = relevance_scores.index(
        max_score) if relevance_scores else -1

    final_knowledge = get_final_knowledge(query, top_chunks, max_score, best_doc_idx)
        
    # 定义系统指令（system prompt），指导模型如何生成回答
    system_prompt = """
    你是一个乐于助人的AI助手。请根据提供的知识内容，生成一个全面且有信息量的回答。
    在回答中包含所有相关信息，同时保持语言清晰简洁。
    如果知识内容不能完全回答问题，请指出这一限制。
    最后在回答末尾注明引用来源。
    """

    # 构建用户提示（user prompt），包含用户的查询、知识内容和来源信息
    user_prompt = f"""
    查询内容：{query}

    知识内容：
    {final_knowledge}

    请根据以上信息，提供一个有帮助的回答，并在最后列出引用来源。
    """
    
    # 7. 调用LLM模型，生成回答
    result = query_llm(system_prompt, user_prompt)
    info(f"--8--->final result: {result}")
