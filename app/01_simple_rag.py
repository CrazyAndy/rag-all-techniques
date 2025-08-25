from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from dotenv import load_dotenv
from utils.llm_utils import query_llm
from utils.similarity_utils import cosine_similarity

load_dotenv()


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


def similar_search(text_chunks, knowledge_base_embeddings, query_embeddings, k=5):
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

    for i, chunk_embedding in enumerate(knowledge_base_embeddings):
        similarity_score = cosine_similarity(chunk_embedding, query_vector)
        similarity_scores.append((i, similarity_score))

    # 按相似度降序排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 返回包含文本块和相似度分数的结果
    results = []
    for index, score in similarity_scores[:k]:
        results.append({
            'text': text_chunks[index],
            'score': score,
            'index': index
        })

    return results


if __name__ == "__main__":

    query = "孙悟空的兵器是什么？"

    # 1. 提取文本
    print("---1--->正在提取西游记文本...")
    extract_text = extract_text_from_markdown()
    # print(f"文本长度: {len(extract_text)} 字符")

    # 2. 分割文本
    print("---2--->正在分割文本...")
    knowledge_chunks = chunk_text(extract_text, 1000, 200)
    # print(f"分割为 {len(text_chunks)} 个文本块")

    # 3. 构建向量数据库
    print("---3--->正在构建向量模型...")
    embedding_model = EmbeddingModel()

    # 4. 将知识库文本块向量化
    print("---4--->正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(knowledge_chunks)

    # 5. 构建问题向量
    print("---5--->正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 6. 向量相似度检索
    print("---6--->向量相似度检索...")
    top_chunks = similar_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, 5)

    print("搜索结果:")
    for i, result in enumerate(top_chunks):
        print(f"{i+1}. 相似度分数: {result['score']:.4f}")
        print(f"   文档: {result['text'][:100]}...")
        print()

    system_prompt = """
    你是一个AI助手，请严格根据以下信息回答问题。如果信息中没有答案，请回答“我不知道”。"""

    user_prompt = "\n".join(
        [f"上下文内容 {i + 1} :\n{result['text']}\n========\n"
         for i, result in enumerate(top_chunks)])

    user_prompt = f"{user_prompt}\n\n Question: {query}"

    # 7. 调用LLM模型，生成回答
    result = query_llm(system_prompt, user_prompt)
    print(f"---7--->final result: {result}")
