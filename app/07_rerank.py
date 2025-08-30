import re
from tqdm import tqdm
from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm, query_llm_with_top_chunks
from utils.similarity_utils import similar_search
from utils.logger_utils import info

# 0. 构建全局向量模型
embedding_model = EmbeddingModel()


def rerank_search_by_query_sentence(top_chunks, query):
    # 定义 LLM 的系统提示
    system_prompt = """
    您是文档相关性评估专家，擅长判断文档与搜索查询的匹配程度。您的任务是根据文档对给定查询的应答质量，给出0到10分的评分。

    评分标准：
    0-2分：文档完全无关
    3-5分：文档含部分相关信息但未直接回答问题
    6-8分：文档相关且能部分解答查询
    9-10分：文档高度相关且直接准确回答问题

    必须仅返回0到10之间的单个整数评分，不要包含任何其他内容。
    """
    scored_results = []  # 初始化一个空列表以存储评分结果
    for i in tqdm(range(len(top_chunks)), desc="根据查询句子和片段的相关性，重新排序"):
        chunk = top_chunks[i]
        # 定义 LLM 的用户提示
        user_prompt = f"""
        查询: {query}

        文档:
        {chunk['text']}

        请对文档的相关性进行评分，评分范围为 0 到 10, 并仅返回一个整数。
        """

        score_text = query_llm(system_prompt, user_prompt)

        # 使用正则表达式提取数值评分
        score_match = re.search(r'\b(10|[0-9])\b', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            # 如果评分提取失败，使用相似度评分作为备选
            print(f"警告：无法从响应中提取评分：'{score_text}'，使用相似度评分代替")
            score = result["similarity"] * 10

        # 将评分结果添加到列表中
        scored_results.append({
            "text": chunk["text"],
            "similarity": chunk["score"],
            "relevance_score": score
        })

    # 按相关性评分降序对结果进行排序
    reranked_results = sorted(
        scored_results, key=lambda x: x["relevance_score"], reverse=True)
    return query_llm_with_top_chunks(reranked_results, query)


def rerank_search_by_query_keywords(top_chunks, query):
    # 定义 LLM 的系统提示
    system_prompt = """
    您是文字语言专家，擅长将语句中的内容提炼出关键词。您的任务是根据[用户询问问题]，提炼出[关键词]。

    要求：
    1. [关键词]要尽可能的包含用户询问的问题
    2. [关键词]要尽可能的简洁
    
    只返回[关键词]列表，并用逗号隔开，不要包含任何其他内容。
    """
    user_prompt = f"""
    [用户询问问题]: {query}
    """
    scored_results = []  # 初始化一个空列表以存储评分结果
    keywords_text = query_llm(system_prompt, user_prompt)
    keywords = keywords_text.split(",")

    for i in tqdm(range(len(top_chunks)), desc="根据查询关键词和片段的相似性，重新排序"):
        chunk = top_chunks[i]
        chunk_text = chunk["text"]
        # 基础分数从向量相似度开始
        base_score = chunk["score"] * 0.5
        # 初始化关键词分数
        keyword_score = 0
        for keyword in keywords:
            if keyword in chunk_text.lower():
                # 每找到一个关键词加一些分数
                keyword_score += 0.1
                # 如果关键词出现在文本开头部分，额外加分
                first_position = chunk_text.find(keyword)
                if first_position < len(chunk_text) / 4:  # 在文本的前四分之一部分
                    keyword_score += 0.1

                # 根据关键词出现的频率加分
                frequency = chunk_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)  # 最大值限制为 0.2

        # 通过结合基础分数和关键词分数计算最终得分
        final_score = base_score + keyword_score

        # 将评分结果添加到列表中
        scored_results.append({
            "text": chunk_text,
            "similarity": base_score,
            "relevance_score": final_score
        })
    # 按相关性评分降序对结果进行排序
    reranked_results = sorted(
        scored_results, key=lambda x: x["relevance_score"], reverse=True)
    return query_llm_with_top_chunks(reranked_results, query)


if __name__ == "__main__":

    query = "孙悟空是跟谁学的72变？"
    info(f"--0--> Question: {query}")
    # 1. 提取文本
    info("--1--> 正在提取西游记文本...")
    extract_text = extract_text_from_markdown()

    # 2. 分割文本
    info("---2--->正在分割文本...")
    # 这里single_chunk_size要是设置成1000，就无法检索到相关内容
    knowledge_chunks = chunk_text_by_length(extract_text, 1000, 200)

    # 3. 将知识库文本块向量化
    info("--3--> 正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(
        knowledge_chunks)

    # 4. 构建问题向量
    info("--4--> 正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. 向量相似度检索
    info("--5--> 语义相似度检索...")
    top_chunks = similar_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, 10)

    info(f"--5--> 搜索结果:")
    for i, result in enumerate(top_chunks):
        info(
            f"  {i+1}. 相似度分数: {result['score']:.4f} ")
        info(f"    文档: {result['text'][:100]}...")

    # 6. 根据查询句子和片段的相关性，重新排序
    info(f"--6--> 根据查询句子和片段的相关性，重新排序")
    answer_by_reranking_sentence = rerank_search_by_query_sentence(
        top_chunks, query)

    info(f"--6--> 根据查询句子和片段的相关性，llm答案：{answer_by_reranking_sentence}")

    # 7. 根据查询句子的关键词和片段的相似性，重新排序 ，这种也可以理解成句子和片段的相关性
    info(f"--7--> 根据查询句子的关键词和片段的相似性，重新排序")
    answer_by_reranking_keywords = rerank_search_by_query_keywords(
        top_chunks, query)

    info(f"--7--> 根据查询句子的关键词和片段的相似性，llm答案: {answer_by_reranking_keywords}")
