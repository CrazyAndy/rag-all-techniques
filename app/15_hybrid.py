from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from dotenv import load_dotenv
from utils.llm_utils import query_llm
from utils.similarity_utils import cosine_similarity, similar_search
from utils.logger_utils import info
import jieba
from rank_bm25 import BM25Okapi
import re
import numpy as np

embedding_model = EmbeddingModel()

def bm25_search(bm25, chunks, query, k=5):
    """
    使用查询在 BM25 索引中进行搜索。

    Args:
        bm25 (BM25Okapi): BM25 索引
        chunks (List[Dict]): 文本块列表
        query (str): 查询字符串
        k (int): 返回的结果数量

    Returns:
        List[Dict]: 带有分数的前 k 个结果
    """
    # 将查询按空格分割成单独的词
    # query_tokens = query.split()  # 英文
    query_tokens = list(jieba.cut(query))   # 中文

    # 获取查询词对已索引文档的 BM25 分数
    scores = bm25.get_scores(query_tokens)

    # 初始化一个空列表，用于存储带有分数的结果
    results = []

    # 遍历分数和对应的文本块
    for i, score in enumerate(scores):
        # 创建元数据的副本以避免修改原始数据
        metadata = chunks[i].get("metadata", {}).copy()
        # 向元数据中添加索引
        metadata["index"] = i

        results.append({
            "text": chunks[i]["text"],  # 文本内容
            "metadata": metadata,  # 带索引的元数据
            "bm25_score": float(score)  # BM25 分数
        })

    # 按 BM25 分数降序排序结果
    results.sort(key=lambda x: x["bm25_score"], reverse=True)

    # 返回前 k 个结果
    return results[:k]

def create_bm25_index(chunks):
    """
    从给定的文本块创建 BM25 索引。

    Args:
        chunks (List[Dict]): 文本块列表

    Returns:
        BM25Okapi: BM25 索引
    """

    # 按空白字符分割对每个文档进行分词
    # tokenized_docs = [text.split() for text in texts]   # 英文
    tokenized_docs = [list(jieba.cut(text)) for text in chunks]  # 中文

    # 使用分词后的文档创建 BM25 索引
    bm25 = BM25Okapi(tokenized_docs)

    # 打印 BM25 索引中的文档数量
    print(f"已创建包含 {len(chunks)} 个文档的 BM25 索引")

    return bm25


def clean_text(text):
    """
    通过移除多余的空白字符和特殊字符来清理文本。

    Args:
        text (str): 输入文本

    Returns:
        str: 清理后的文本
    """
    # 将多个空白字符（包括换行符和制表符）替换为一个空格
    text = re.sub(r'\s+', ' ', text)

    # 修复常见的OCR问题，将制表符和换行符替换为空格
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')

    # 移除开头和结尾的空白字符，并确保单词之间只有一个空格
    text = ' '.join(text.split())

    return text


def create_score_for_knowledge_chunks(query,query_embedding,query_tokens,knowledge_chunks,knowledge_embeddings,bm25_index,k=8,alpha=0.5):
    # 确保查询向量是一维的
    if isinstance(query_embeddings, list):
        query_vector = query_embeddings[0]
    elif hasattr(query_embeddings, 'shape') and len(query_embeddings.shape) > 1:
        query_vector = query_embeddings[0]
    else:
        query_vector = query_embeddings
        
    # 获取查询词对已索引文档的 BM25 分数
    bm25_scores = bm25_index.get_scores(query_tokens)
    
    combined_results = []
    
    similarity_scores = []    
    for i,chunk in enumerate(knowledge_chunks):
        chunk_embedding = knowledge_embeddings[i]
        similarity_score = cosine_similarity(chunk_embedding, query_vector)
        bm25_score = float(bm25_scores[i])
        similarity_scores.append(similarity_score)
        combined_results.append({
            "text": chunk,
            "similarity_score": similarity_score,
            "bm25_score": bm25_score,
            "index": i
        })
        
        
    # 定义一个小的 epsilon 来避免除以零
    epsilon = 1e-8
    similarity_score_width = np.max(similarity_scores) - np.min(similarity_scores) + epsilon
    bm25_score_width = np.max(bm25_scores) - np.min(bm25_scores) + epsilon
    similarity_score_min = np.min(similarity_scores)
    bm25_score_min = np.min(bm25_scores)
    
    for result in combined_results:
        norm_similarity_score = (result["similarity_score"] - similarity_score_min) / similarity_score_width
        norm_bm25_score = (result["bm25_score"] - bm25_score_min) / bm25_score_width
        combined_score = alpha * norm_similarity_score + (1 - alpha) * norm_bm25_score
        result["combined_score"] = combined_score
    
    # 按综合分数排序（降序）
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

    # 返回前 k 个结果
    return combined_results[:k]


if __name__ == "__main__":

    query = "孙悟空的兵器是什么？"

    # 1. 提取文本
    info("---1--->正在提取西游记文本...")
    extract_text = extract_text_from_markdown()
    # print(f"文本长度: {len(extract_text)} 字符")
    
    # 清理提取的文本，去除多余的空白和特殊字符
    cleaned_text = clean_text(extract_text)

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

    # 5. 构建 BM25 索引
    info("---5--->正在构建 BM25 索引...")
    bm25_index = create_bm25_index(knowledge_chunks)
    
    # 6. 将问题按空格分割成单独的词
    info("---6--->将问题按空格分割成单独的词...")
    query_tokens = list(jieba.cut(query))   # 中文
    
    # 7. 创建综合评分
    info("---7--->创建综合评分...")
    top_chunks = create_score_for_knowledge_chunks(query,query_embeddings,query_tokens,knowledge_chunks,knowledge_embeddings,bm25_index,k=5,alpha=0.5)
    

    info("搜索结果:")
    for i, result in enumerate(top_chunks):
        info(f"{i+1}. 相似度分数: {result['combined_score']:.4f}")
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
    info(f"--8--->final result: {result}")

