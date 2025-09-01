
import re
from tqdm import tqdm
from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm, query_llm_with_top_chunks
from utils.similarity_utils import similar_search
from utils.logger_utils import info

# 0. 构建全局向量模型
embedding_model = EmbeddingModel()


def find_best_segments_sorted_greedy(knowledge_chunk_values, top_k=5, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    """
    排序贪心算法 (Sorted Greedy Algorithm)

    算法步骤：
    1. 生成所有可能的段落候选
    2. 按得分降序排序
    3. 贪心选择非重叠段落

    时间复杂度: O(n × max_segment_length × log(n × max_segment_length))
    空间复杂度: O(n × max_segment_length)

    Args:
        chunk_values (List[float]): 每个块的值
        max_segment_length (int): 单个段落的最大长度
        total_max_length (int): 所有段落的最大总长度
        min_segment_value (float): 被考虑的段落的最小值

    Returns:
        List[Tuple[int, int]]: 最佳段落的（开始，结束）索引列表
    """

    n = len(knowledge_chunk_values)

    # 步骤1: 生成候选段落 - O(n × max_segment_length)
    candidates = []
    for start in range(n):
        for length in range(1, min(max_segment_length, n - start) + 1):
            end = start + length
            segment_sum = sum(knowledge_chunk_values[start:end])
            segment_avg = segment_sum / length

            # 评分函数：总得分 + 密度奖励
            score = segment_sum + segment_avg * 0.5

            if score >= min_segment_value:
                candidates.append((score, start, end))

    # 步骤2：重新排序候选段落
    # 如果不排序，算法会按照生成顺序（通常是按起始位置和长度）来选择段落
    # 可能会先选择得分较低的段落，占用位置后，得分更高的段落因为重叠而被拒绝
    # 最终选择的段落组合不是最优的            
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 步骤3: 贪心选择非重叠段落 - O(n × max_segment_length)
    best_segments = []
    total_included_chunks = 0
    used_positions = set()

    for score, start, end in candidates:
        if total_included_chunks >= total_max_length:
            break

        # 检查重叠
        overlap = False
        for pos in range(start, end):
            if pos in used_positions:
                overlap = True
                break

        if not overlap:
            best_segments.append((start, end, score))
            total_included_chunks += end - start

            # 标记已使用位置
            for pos in range(start, end):
                used_positions.add(pos)

#             print(f"选择段落 ({start}, {end})，得分 {score:.4f}")

    # 按照得分从高到低排序
    best_segments.sort(key=lambda x: x[2], reverse=True)

    return best_segments[0:min(top_k, len(best_segments))]


def calculate_chunk_values(knowledge_chunks, top_chunks):
    # 计算块值（相关性分数减去惩罚）
    chunk_values = []
    for index in range(len(knowledge_chunks)):
        # 获取相关性分数，如果不在结果中则默认为0,如果top_chunks中index字段等于index则相关性分数为top_chunks中score字段
        for top_chunk in top_chunks:
            if top_chunk['index'] == index:
                score = top_chunk['score']
                break
        else:
            score = -0.2  # 应用惩罚以将不相关的块转换为负值
        # 将相关性分数添加到块值列表中
        chunk_values.append(score)
    return chunk_values


def reconstruct_chunks(best_segments, knowledge_chunks):
    reconstructed_chunks = []
    for segment in best_segments:
        reconstructed_chunks.append({
            "text": " ".join(knowledge_chunks[segment[0]:segment[1]]),
            "score": segment[2]
        })
    return reconstructed_chunks


if __name__ == "__main__":

    query = "猪八戒想取的媳妇是谁？"
    info(f"--0--> Question: {query}")
    # 1. 提取文本
    info("--1--> 正在提取西游记文本...")
    extract_text = extract_text_from_markdown()

    # 2. 分割文本
    info("---2--->正在分割文本...")
    knowledge_chunks = chunk_text_by_length(extract_text, 500, 0)

    info(f"--------------------------------")
    info(f"分割后的文本块数量: {len(knowledge_chunks)}")
    info(f"--------------------------------")

    # 3. 将知识库文本块向量化
    info("--3--> 正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(
        knowledge_chunks)

    # 4. 构建问题向量
    info("--4--> 正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. 向量相似度检索
    info("--5--> 语义相似度检索...")
    top_chunks = similar_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, 10)

    # 6. 计算所有的块的相似度评分值
    info("--6--> 计算所有的块的相似度评分值...")
    knowledge_chunk_values = calculate_chunk_values(
        knowledge_chunks, top_chunks)

    # 7. 根据得分从高到低排序，找出最佳的段落
    info("--7--> 根据得分从高到低排序，找出最佳的段落...")
    best_segments = find_best_segments_sorted_greedy(
        knowledge_chunk_values, 5, 20, 30, 0.2)

    # 8. 根据最佳的段落，重新构建文本
    info(f"--8--> 根据最佳的段落，重新构建文本...{len(best_segments)}")
    reconstructed_chunks = reconstruct_chunks(best_segments, knowledge_chunks)

    info(f"--8--> 重新构建的文本:")
    for i, result in enumerate(reconstructed_chunks):
        info(
            f"  {i+1}. 相似度分数: {result['score']:.4f} ")
        info(f"    文档: {result['text'][:200]}...\n")

    # 9. 根据最佳的段落，结合问题给大模型进行回答
    info("--9--> 根据最佳的段落，结合问题给大模型进行回答...\n")
    answer = query_llm_with_top_chunks(reconstructed_chunks, query)
    info(f"大模型最终答案: {answer}")
