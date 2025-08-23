from utils.file_utils import extract_text_from_markdown
from utils.chroma_utils import create_chroma_db
from dotenv import load_dotenv
import os

from utils.llm_utils import query_llm

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


def build_vector_database():
    """
    构建向量数据库

    Args:
        text_chunks (list): 文本块列表

    Returns:
        ChromaVectorDB: 向量数据库实例
    """
    # 创建 Chroma 向量数据库
    chroma_db = create_chroma_db()
    chroma_db.delete_collection()
    return chroma_db


def create_embeddings(chroma_db, text_chunks):
    embeddings = chroma_db.create_embeddings(text_chunks)
    return embeddings


def semantic_search(chroma_db, query, text_chunks, embeddings, k=2):
    # 添加文本块到向量数据库
    chroma_db.add_texts(text_chunks, embeddings)

    # 生成查询嵌入向量
    query_embedding = chroma_db.create_embeddings([query])

    # 执行语义搜索
    results = chroma_db.search(query_embedding, k)

    return results


if __name__ == "__main__":
    # 1. 提取文本
    print("正在提取西游记文本...")
    extract_text = extract_text_from_markdown()
    # print(f"文本长度: {len(extract_text)} 字符")

    # 2. 分割文本
    print("正在分割文本...")
    text_chunks = chunk_text(extract_text, 1000, 200)
    # print(f"分割为 {len(text_chunks)} 个文本块")

    # 3. 构建向量数据库
    print("正在构建向量数据库...")
    chroma_db = build_vector_database()

    # 4. 显示数据库信息
    embeddings = create_embeddings(chroma_db, text_chunks)
    # print(f"embeddings: {embeddings}")

    # 5. 测试搜索功能
    print("\n测试搜索功能...")
    test_query = "孙悟空的兵器是什么？"
    top_chunks = semantic_search(
        chroma_db, test_query, text_chunks, embeddings, 10)

    print(f"查询: '{test_query}'")
    print("搜索结果:")
    for i, (doc, metadata) in enumerate(zip(top_chunks['documents'][0], top_chunks['metadatas'][0])):
        print(f"{i+1}. 相似度: {top_chunks['distances'][0][i]:.4f}")
        print(f"   文档: {doc[:100]}...")
        print(f"   元数据: {metadata}")
        print()
        
    system_prompt = """
    你是一个AI助手，请严格根据以下信息回答问题。如果信息中没有答案，请回答“我不知道”。"""
    
    user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n========\n" for i, chunk in enumerate(top_chunks)])
    user_prompt = f"{user_prompt}\nQuestion: {test_query}"
    
    result = query_llm(system_prompt, user_prompt)
    print(f"final result: {result}")
        
    
        
    
