from utils.file_utils import extract_text_from_markdown
from utils.logger_utils import info



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
    chunks = []
    for i in range(0, len(text), single_chunk_size - overlap):
        chunks.append(text[i:i + single_chunk_size])
    return chunks




if __name__ == "__main__":

    query = "孙悟空的兵器是从哪里来的？"
    info(f"--0--> Question: {query}")

    # 1. 提取文本
    info("--1--> 正在提取西游记文本...")
    extract_text = extract_text_from_markdown()

    # 2. 分割文本
    info("---2--->正在分割文本...")
    knowledge_chunks = chunk_text(extract_text, 2000, 200) # 这里single_chunk_size要是设置成1000，就无法检索到相关内容

    # 3. 将知识库文本块向量化
    info("--3--> 正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(
        knowledge_chunks, show_progress=True)
    print("")
    # 4. 构建问题向量
    info("--4--> 正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. 向量相似度检索
    info("--5--> 语义相似度检索...")
    top_chunks = context_enriched_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, 5)

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