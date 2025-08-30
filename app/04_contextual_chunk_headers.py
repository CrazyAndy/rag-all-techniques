from tqdm import tqdm
from utils.common_utils import validate_chunk_text_params
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm
from utils.logger_utils import info
from utils.similarity_utils import cosine_similarity

# 0. 构建全局向量模型
embedding_model = EmbeddingModel()


def chunk_text_with_headers(text, single_chunk_size, overlap):
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
    # 参数校验
    validate_chunk_text_params(text, single_chunk_size, overlap)
    
    system_prompt = "为给定的文本生成一个简洁且信息全面的标题。"
    chunks = []
    for i in tqdm(range(0, len(text), single_chunk_size - overlap), desc="chunk_text_with_headers"):
        chunk = text[i:i + single_chunk_size]
        header = query_llm(system_prompt, chunk)  # 使用 LLM 为块生成标题
        chunks.append({"header": header, "text": chunk})  # 将标题和块添加到列表中
    return chunks


def create_embeddings_for_knowledge_chunks(knowledge_chunks):
    '''
    将知识库文本块向量化
    '''
    knowledge_embeddings = []  # Initialize an empty list to store embeddings

    # Iterate through each text chunk with a progress bar
    for chunk in tqdm(knowledge_chunks, desc="Generating embeddings"):
        # Create an embedding for the chunk's text
        text_embedding = embedding_model.create_embeddings(chunk["text"])
        # print(text_embedding.shape)
        # Create an embedding for the chunk's header
        header_embedding = embedding_model.create_embeddings(chunk["header"])
        # Append the chunk's header, text, and their embeddings to the list
        knowledge_embeddings.append({"header": chunk["header"], "text": chunk["text"], "content_embedding": text_embedding,
                                     "header_embedding": header_embedding})

    return knowledge_embeddings


def similar_search(knowledge_chunks,knowledge_embeddings, query_embeddings, k=5):
    similarity_scores = []
    # 确保查询向量是一维的
    if isinstance(query_embeddings, list):
        query_vector = query_embeddings[0]
    elif hasattr(query_embeddings, 'shape') and len(query_embeddings.shape) > 1:
        query_vector = query_embeddings[0]
    else:
        query_vector = query_embeddings

    for chunk_index, chunk in enumerate(knowledge_embeddings):
        # 计算查询嵌入与当前文本块嵌入之间的余弦相似度
        similarity_score_content = cosine_similarity(
            chunk["content_embedding"], query_vector)
        # 计算查询嵌入与当前文本块标题嵌入之间的余弦相似度
        similarity_score_header = cosine_similarity(
            chunk["header_embedding"], query_vector)
        # 计算平均相似度分数
        avg_similarity = (similarity_score_content +
                          similarity_score_header) / 2
        similarity_scores.append((chunk_index, avg_similarity))

    # 按相似度分数降序排序（相似度最高排在前面）
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
     # 返回包含文本块和相似度分数的结果
    results = []
    for index in range(min(k, len(similarity_scores))):
        chunk_index, chunk_score = similarity_scores[index]
        results.append({
            'text': knowledge_chunks[chunk_index]["text"],
            'score': chunk_score,
            'index': chunk_index
        })

    return results


if __name__ == "__main__":

    query = "孙悟空的兵器是从哪里来的？"
    info(f"--0--> Question: {query}")

    # 1. 提取文本
    info("--1--> 正在提取西游记文本...")
    extract_text = extract_text_from_markdown()

    # 2. 分割文本
    info("---2--->正在分割文本...")
    # 这里single_chunk_size要是设置成1000，就无法检索到相关内容
    knowledge_chunks = chunk_text_with_headers(extract_text, 1000, 200)

    # 3. 将知识库文本块向量化
    info("--3--> 正在构建知识库向量集...")
    knowledge_embeddings = create_embeddings_for_knowledge_chunks(
        knowledge_chunks)

    # 4. 构建问题向量
    info("--4--> 正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. 向量相似度检索
    info("--5--> 语义相似度检索...")
    top_chunks = similar_search(knowledge_chunks,knowledge_embeddings, query_embeddings, 5)

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
