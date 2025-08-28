from tqdm import tqdm
from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm
from utils.logger_utils import info
from utils.similarity_utils import cosine_similarity
import re

# 0. 构建全局向量模型
embedding_model = EmbeddingModel()


def generate_questions(text_chunk, num_questions=5):
    """
    生成可以从给定文本块中回答的相关问题。

    Args:
    text_chunk (str): 要生成问题的文本块。
    num_questions (int): 要生成的问题数量。
    model (str): 用于生成问题的模型。

    Returns:
    List[str]: 生成的问题列表。
    """
    # 定义系统提示
    system_prompt = "你是一个从文本中生成相关问题的专家。能够根据用户提供的文本生成可回答的简洁问题，重点聚焦核心信息和关键概念。"

    user_prompt = f"""
    请根据以下文本内容生成{num_questions}个不同的、仅能通过该文本内容回答的问题：
    
    ---文本内容开始---
    {text_chunk}
    ---文本内容结束---
    
    # 限制
    请严格按以下格式回复：
    1. 带编号的问题列表
    2. 仅包含问题
    3. 不要添加任何其他内容
    """
    # 使用 OpenAI API 生成问题
    response = query_llm(system_prompt, user_prompt)

    # 使用正则表达式模式匹配提取问题
    pattern = r'^\d+\.\s*(.*)'
    return [re.match(pattern, line).group(1) for line in response.split('\n') if line.strip()]


def create_embeddings_for_knowledge_chunks(knowledge_chunks, question_count_per_chunk=3):
    '''
    将知识库文本块向量化
    '''
    knowledge_data = []  # Initialize an empty list to store embeddings

    questions_data = []  # 单个chunk生成多个问题

    # Iterate through each text chunk with a progress bar
    for index in tqdm(range(len(knowledge_chunks)), desc="创建知识库和问题库数据集"):
        chunk = knowledge_chunks[index]
        # Create embeddings for each chunk
        knowledge_data.append({
            "chunk_index": index,
            "chunk_text": chunk,
            "chunk_embedding": embedding_model.create_embeddings(chunk)
        })

        # 为该文本块生成问题
        questions = generate_questions(
            chunk, num_questions=question_count_per_chunk)
        for question in questions:
            questions_data.append({
                "chunk_index": index,
                "chunk_text": chunk,
                "question_text": question,
                "question_embedding": embedding_model.create_embeddings(question)
            })

    return knowledge_data, questions_data


def semantic_search(query_embeddings, knowledge_data, questions_data, k=5):
    similarity_scores = []
    # 确保查询向量是一维的
    if isinstance(query_embeddings, list):
        query_vector = query_embeddings[0]
    elif hasattr(query_embeddings, 'shape') and len(query_embeddings.shape) > 1:
        query_vector = query_embeddings[0]
    else:
        query_vector = query_embeddings

    for single_data in knowledge_data:
        # 计算查询嵌入与当前文本块嵌入之间的余弦相似度
        similarity_score = cosine_similarity(
            single_data["chunk_embedding"], query_vector)
        similarity_scores.append(
            (single_data["chunk_index"], single_data["chunk_text"], similarity_score))

    for single_data in questions_data:
        # 计算查询嵌入与当前文本块嵌入之间的余弦相似度
        similarity_score = cosine_similarity(
            single_data["question_embedding"], query_vector)
        similarity_scores.append(
            (single_data["chunk_index"], single_data["chunk_text"], similarity_score))

    # 按相似度分数降序排序（相似度最高排在前面）
    similarity_scores.sort(key=lambda x: x[2], reverse=True)

    final_similarity_scores = []
    for index, text, score in similarity_scores:
        if len(final_similarity_scores) == k:
            break
        # 检查是否已经存在相同index的元素，如果存在则跳过
        if any(item["index"] == index for item in final_similarity_scores):
            continue

        final_similarity_scores.append({
            "index": index,
            "text": text,
            "score": score
        })

    # Return the top-k most relevant chunks
    return final_similarity_scores


if __name__ == "__main__":

    query = "铁扇公主的芭蕉扇借给孙悟空了吗？"
    info(f"--0--> Question: {query}")

    # 1. 提取文本
    info("--1--> 正在提取西游记文本...")
    extract_text = extract_text_from_markdown()

    # 2. 分割文本
    info("---2--->正在分割文本...")
    # 这里single_chunk_size要是设置成1000，就无法检索到相关内容
    knowledge_chunks = chunk_text_by_length(extract_text, 1000, 200)

    # 3. 将知识库文本块向量化
    info("--3--> 正在构建知识库向量集，chunk生成问题向量集...")
    knowledge_data, questions_data = create_embeddings_for_knowledge_chunks(
        knowledge_chunks)

    # 4. 构建问题向量
    info("--4--> 正在构建用户问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. 向量相似度检索
    info("--5--> 语义相似度检索...")
    top_chunks = semantic_search(
        query_embeddings, knowledge_data, questions_data, 5)

    info(f"--5--> 搜索结果:")
    for i, result in enumerate(top_chunks):
        info(
            f"  {i+1}. 相似度分数: {result['score']:.4f} ")
        info(f"    文档: {result['text'][:100]}...")

    system_prompt = """
    你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。"""

    user_prompt = "\n".join(
        [f"上下文内容 {i + 1} :\n{result["text"]}\n========\n"
         for i, result in enumerate(top_chunks)])

    user_prompt = f"{user_prompt}\n\n Question: {query}"

    # 7. 调用LLM模型，生成回答
    result = query_llm(system_prompt, user_prompt)
    info(f"--6--> final result: {result}")
