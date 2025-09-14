from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm
from utils.similarity_utils import similar_search
from utils.logger_utils import info


embedding_model = EmbeddingModel()


def generate_hypothetical_document(query, desired_length=1000):
    """
    生成能够回答查询的假设文档

    Args:
        query (str): 用户查询内容
        desired_length (int): 目标文档长度（字符数）

    Returns:
        str: 生成的假设文档文本
    """
    # 定义系统提示词以指导模型生成文档的方法
    system_prompt = f"""你是一位专业的文档创建专家。
    给定一个问题，请生成一份能够直接解答该问题的详细文档。
    文档长度应约为 {desired_length} 个字符，需提供深入且具有信息量的答案。
    请以权威资料的口吻撰写，内容需包含具体细节、事实和解释。
    不要提及这是假设性文档 - 直接输出内容即可。"""

    # 用查询定义用户提示词
    user_prompt = f"问题: {query}\n\n生成一份完整解答该问题的文档："

    # 调用OpenAI API生成假设文档
    return query_llm(system_prompt, user_prompt)


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

    # 4. 生成一个假设文档来回答查询
    info("---4--->正在生成假设文档...")
    hypothetical_doc = generate_hypothetical_document(query)
    info(f"生成了长度为 {len(hypothetical_doc)} 个字符的假设文档")
    info(f"假设文档: {hypothetical_doc}")

    # 5. 构建问题向量
    info("---5--->正在构建假设文档向量...")
    hypothetical_embedding = embedding_model.create_embeddings(
        [hypothetical_doc])

    # 6. 向量相似度检索
    info("---6--->向量相似度检索...")
    top_chunks = similar_search(
        knowledge_chunks, knowledge_embeddings, hypothetical_embedding, 5)

    info("搜索结果:")
    for i, result in enumerate(top_chunks):
        info(f"{i+1}. 相似度分数: {result['score']:.4f}")
        info(f"   文档: {result['text'][:100]}...")
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
