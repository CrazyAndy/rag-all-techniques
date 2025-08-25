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
    # Compute the dot product of the two vectors and divide by the product of their norms
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
