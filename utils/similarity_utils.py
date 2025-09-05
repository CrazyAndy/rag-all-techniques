import numpy as np


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度，返回一个相似度值

    其实就是计算两个向量夹角的余弦值，值越大，相似度越高

    Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    
    # 转换为numpy数组并确保是一维的
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()
    
    # Compute the dot product of the two vectors and divide by the product of their norms
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def similar_search(knowledge_chunks, knowledge_base_embeddings, query_embeddings, k=5):
    """
    语义搜索，计算相似度并返回最相关的文本块

    Args:
        knowledge_chunks: 知识库文本块列表
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
    for index in range(min(k, len(similarity_scores))):
        chunk_index, chunk_score = similarity_scores[index]
        results.append({
            'text': knowledge_chunks[chunk_index],
            'score': chunk_score,
            'index': chunk_index
        })

    return results
