from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from dotenv import load_dotenv
from utils.llm_utils import query_llm
from utils.similarity_utils import similar_search
from utils.logger_utils import info

load_dotenv()


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

    # 3. 构建向量数据库
    info("---3--->正在构建向量模型...")
    embedding_model = EmbeddingModel()

    # 4. 将知识库文本块向量化
    info("---4--->正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(knowledge_chunks)

    # 5. 构建问题向量
    info("---5--->正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 6. 向量相似度检索
    info("---6--->向量相似度检索...")
    top_chunks = similar_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, 5)

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
