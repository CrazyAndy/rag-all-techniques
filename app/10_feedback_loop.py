import re
from typing import List
from tqdm import tqdm
from utils.common_utils import chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm
from utils.similarity_utils import cosine_similarity, similar_search
from utils.logger_utils import info
import json


# 0. 构建全局向量模型
embedding_model = EmbeddingModel()

feedback_data_file_path = "feedback_data.json"


def load_feedback_data():
    """
    从文件中加载反馈数据。

    Args:
        feedback_file (str): 反馈文件的路径

    Returns:
        List[Dict]: 反馈条目的列表
        这里记录了 问题，答案，用户评分相关度，评分质量
    """
    feedback_data = []
    try:
        with open(feedback_data_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        feedback_item = json.loads(line)
                        feedback_data.append(feedback_item)
                    except json.JSONDecodeError as e:
                        print(f"第{line_num}行JSON解析错误: {e}")
                        print(f"问题行内容: {line[:100]}...")
                        continue  # 跳过有问题的行，继续处理其他行

    except FileNotFoundError:
        print("未找到反馈数据文件。将以空反馈开始。")
    except Exception as e:
        print(f"读取反馈数据时出错: {e}")

    return feedback_data


def store_feedback(feedback):
    """
    将反馈存储在JSON文件中。
    相关度 评分是从1到5分
    1表示问题和答案完全不相关，最不符合用户预期
    5表示问题和答案完全相关，最符合用户预期
    质量评分是从1到5分
    1表示对于问题来说，答案质量最低，无法回答问题
    5表示对于问题来说，答案质量最高，完全符合用户预期
    Args:
        feedback (Dict): 反馈数据
        feedback_file (str): 反馈文件的路径
    """
    try:
        # 直接追加新的一行，不需要读取整个文件
        with open(feedback_data_file_path, "a", encoding="utf-8") as f:
            json.dump(feedback, f, ensure_ascii=False)
            f.write("\n")

    except Exception as e:
        print(f"存储反馈数据时出错: {e}")


def clean_question_line(line: str) -> str:
    """
    清理问题行，提取纯问题文本

    Args:
        line: 原始行文本

    Returns:
        清理后的问题文本
    """
    if not line or not line.strip():
        return ""

    # 1. 移除序号和前缀
    # 移除数字序号：1. 1、 1) 等
    line = re.sub(r'^\d+[\.、\)）]\s*', '', line)

    # 移除中文序号：一、二、三、等
    line = re.sub(r'^[一二三四五六七八九十]+[\.、\)）]\s*', '', line)

    # 移除字母序号：a. A、 a) 等
    line = re.sub(r'^[a-zA-Z][\.、\)）]\s*', '', line)

    # 移除罗马数字序号：I. II. III. 等
    line = re.sub(r'^[IVX]+[\.、\)）]\s*', '', line)

    # 2. 移除常见的无关前缀
    prefixes_to_remove = [
        r'^问题\d*[：:]\s*',      # "问题1：" "问题：" 等
        r'^相关[问题问句][：:]\s*',  # "相关问题：" 等
        r'^扩展[问题问句][：:]\s*',  # "扩展问题：" 等
        r'^变体\d*[：:]\s*',      # "变体1：" 等
        r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*',  # 中文圆圈数字
        r'^[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]\s*',  # 中文括号数字
    ]

    for prefix in prefixes_to_remove:
        line = re.sub(prefix, '', line)

    # 3. 移除开头的冒号和空格
    line = re.sub(r'^[：:]\s*', '', line)

    # 4. 清理多余的空格
    line = re.sub(r'\s+', ' ', line).strip()

    # 5. 确保以问号结尾
    if not line.endswith('？'):
        # 如果以其他标点结尾，替换为问号
        if line and line[-1] in '。！，；：':
            line = line[:-1] + '？'
        else:
            line += '？'

    # 6. 最终清理和验证
    line = line.strip()

    # 如果清理后为空或太短，返回空字符串
    if not line or len(line) < 3:
        return ""

    return line


def parse_expanded_questions(llm_response: str) -> List[str]:
    """
    解析大模型返回的扩展问题响应

    Args:
        llm_response: 大模型返回的文本响应

    Returns:
        扩展问题列表
    """
    questions = []

    # 方法1：按行分割，查找包含"？"的行
    lines = llm_response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and '？' in line:
            # 清理行内容
            question = clean_question_line(line)
            if question:
                questions.append(question)

    # 确保至少返回原始问题
    if not questions:
        questions = ["原始问题"]  # 占位符

    return questions[:10]  # 最多返回10个问题


def expand_question_with_llm(original_question: str, original_answer: str) -> List[str]:
    """
    使用大模型扩展问题，生成语义相关的问题变体
    """
    system_prompt = """
    你是一个精通中文语义和表达的专家，请根据用户提供的原始问题和答案，生成10个语义相关但表达不同的问题变体。
    """

    user_prompt = f"""
    要求：
    1. 新问题必须与原始问题语义相同
    2. 使用不同的词汇和句式
    3. 保持问题的核心含义不变
    4. 每个问题都要能通过原始答案回答
    
    原始问题：{original_question}
    原始答案：{original_answer}
    
    请生成10个相关问题，用\n隔开，不要输出其他无关内容
    """

    try:
        response = query_llm(system_prompt, user_prompt)
        info(f"扩展问题时大模型返回: {response}")
        # 解析响应，提取问题列表
        expanded_questions = parse_expanded_questions(response)
        return expanded_questions
    except Exception as e:
        info(f"扩展问题时出错: {e}")
        return [original_question]  # 失败时返回原问题


'''
feedback_embeddings_data是类似如下的json数组
[{
    "original_query": feedback_original_question,
    "original_query_embedding": embedding_model.create_embeddings([feedback_original_question]),
    "original_response_embedding": embedding_model.create_embeddings([feedback['response']]),
    "quality": feedback['quality'],
    "original_related_chunks_embeddings": [embedding_model.create_embeddings([chunk['text']]) for chunk in feedback['related_chunks']],
    "expanded_question_infos": expanded_question_infos,
}]

expanded_question_infos 是类似如下的json数组
[{
    "question": expanded_question,
    "question_embedding": embedding_model.create_embeddings([expanded_question]),
}]

'''


def calculate_expanded_similarity(user_query_embedding, feedback_embeddings_data):
    """
    计算用户问题与扩展问题集的相似度
    """
    max_similarity = 0
    best_match_original_question = ""
    best_match_question = ""
    best_match_feedback_index = -1

    for feedback_index, feedback in enumerate(feedback_embeddings_data):
        feedback_query = feedback['original_query']
        feedback_query_embedding = feedback['original_query_embedding']
        expanded_question_infos = feedback['expanded_question_infos']

        # 与原始问题比较
        original_similarity = cosine_similarity(
            user_query_embedding, feedback_query_embedding)

        max_similarity = original_similarity
        best_match_question = feedback_query

        for expanded_question_info in expanded_question_infos:
            expanded_question = expanded_question_info['question']
            expanded_question_embedding = expanded_question_info['question_embedding']

            expanded_similarity = cosine_similarity(
                user_query_embedding, expanded_question_embedding)

            if expanded_similarity > max_similarity:
                max_similarity = expanded_similarity
                best_match_question = expanded_question
                best_match_feedback_index = feedback_index
                best_match_original_question = feedback_query

    return max_similarity, best_match_question, best_match_feedback_index, best_match_original_question


'''
feedback_embeddings_data是类似如下的json数组
[{
    "original_query": feedback_original_question,
    "original_query_embedding": embedding_model.create_embeddings([feedback_original_question]),
    "original_response_embedding": embedding_model.create_embeddings([feedback['response']]),
    "quality": feedback['quality'],
    "original_related_chunks_embeddings": [embedding_model.create_embeddings([chunk['text']]) for chunk in feedback['related_chunks']],
    "expanded_question_infos": expanded_question_infos,
}]
'''


def calculate_score(query, base_score, user_query_embedding, current_chunk_embedding, feedback_embeddings_data, alpha=0.3, query_threshold=0.7, feedback_threshold=0.7):
    feedback_quality_scores = []

    max_similarity, best_match_question, best_match_feedback_index, best_match_original_question = calculate_expanded_similarity(
        user_query_embedding, feedback_embeddings_data)

    if best_match_feedback_index == -1:
        return base_score

    if max_similarity > query_threshold:
        info(f"当前query:{query}和用户反馈的query:{best_match_original_question}相似度比较高,反馈问题变种是{best_match_question}，相似度为{max_similarity}")
        # 这里要拿当前chunk和feedback中的chunk做相似度对比
        # 拿当前chunk和feedback中的answer作对比
        chunk_similarity = max(cosine_similarity(current_chunk_embedding, old_chunk_embedding)
                               for old_chunk_embedding in feedback_embeddings_data[best_match_feedback_index]['original_related_chunks_embeddings'])
        answer_similarity = cosine_similarity(
            current_chunk_embedding, feedback_embeddings_data[best_match_feedback_index]["original_response_embedding"])

        single_feedback_score = chunk_similarity*0.7 + answer_similarity*0.3

        if single_feedback_score > feedback_threshold:
            info(f"当前chunk和用户反馈的信息总体相似度比较高，相似度为{single_feedback_score}")
            # 说明相似度比较高，我们要拿quality进行操作
            feedback_quality_scores.append(
                feedback_embeddings_data[best_match_feedback_index]['quality'])

    final_feedback_score = 0
    if not feedback_quality_scores:
        return base_score

    if feedback_quality_scores:
        feedback_quality_score_average = sum(
            feedback_quality_scores) / len(feedback_quality_scores)
        info(
            f"feedback_quality_score_average:{feedback_quality_score_average}")
        # 如果评分比较高，这正值，如果评分比较高，则负值
        final_feedback_score = (feedback_quality_score_average - 3)*0.8
        if final_feedback_score == 0:
            return base_score
        final_score = (1 - alpha) * base_score + alpha * final_feedback_score
        return final_score


'''
feedback_data_array是类似如下的json数组
[{
    "query": query,
    "response": result,
    "quality": int(quality),
    "related_chunks": new_top_chunks,
    "expanded_questions": expanded_questions
}]
'''


def optimize_top_chunks_score(query, top_chunks, feedback_data_array):
    if not feedback_data_array:
        return top_chunks

    user_query_embedding = embedding_model.create_embeddings([query])
    feedback_embeddings_data = []  # 这里存储了feedback中每个query,text,response 的向量
    for feedback in feedback_data_array:
        feedback_original_question = feedback['query']

        expanded_question_infos = []
        for question in feedback['expanded_questions']:
            expanded_question_infos.append({
                "question": question,
                "question_embedding": embedding_model.create_embeddings([question])
            })

        feedback_embeddings_data.append({
            "original_query": feedback_original_question,
            "original_query_embedding": embedding_model.create_embeddings([feedback_original_question]),
            "original_response_embedding": embedding_model.create_embeddings([feedback['response']]),
            "quality": feedback['quality'],
            "original_related_chunks_embeddings": [embedding_model.create_embeddings([chunk['text']]) for chunk in feedback['related_chunks']],
            "expanded_question_infos": expanded_question_infos,
        })

    for i in tqdm(range(len(top_chunks))):
        chunk = top_chunks[i]
        base_score = chunk['score']
        current_chunk_embedding = embedding_model.create_embeddings([
                                                                    chunk['text']])
        info(f"\n")
        final_chunk_score = calculate_score(
            query, base_score, user_query_embedding, current_chunk_embedding, feedback_embeddings_data)
        top_chunks[i]['score'] = final_chunk_score

        if base_score != final_chunk_score:
            info(f"单个chunk因为使用用户反馈，评分得到更改，由{base_score}变为{final_chunk_score}")
        else:
            info("当前chunk没有和用户反馈数据中匹配到")

        info(f"\n")

    # 按相似度降序排序
    top_chunks.sort(key=lambda x: x['score'], reverse=True)

    return top_chunks[:5]


if __name__ == "__main__":

    query = "孙悟空会多少种变化的法术？"

    info(f"--0--> Question: {query}")
    # 1. 提取文本
    info("---1--->正在提取西游记文本...")
    extract_text = extract_text_from_markdown()
    # print(f"文本长度: {len(extract_text)} 字符")

    # 2. 分割文本
    info("---2--->正在分割文本...")
    knowledge_chunks = chunk_text_by_length(extract_text, 2000, 200)
    # print(f"分割为 {len(text_chunks)} 个文本块")

    # 3. 将知识库文本块向量化
    info("---3--->正在构建知识库向量集...")
    knowledge_embeddings = embedding_model.create_embeddings(knowledge_chunks)

    # 4. 构建问题向量
    info("---4--->正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 5. 向量相似度检索，这里放大成20个，为后续反馈数据打分做准备
    info("---5--->向量相似度检索...")
    top_chunks = similar_search(
        knowledge_chunks, knowledge_embeddings, query_embeddings, 10)

    info("---5-->搜索结果:")
    for i, result in enumerate(top_chunks):
        info(f" {i+1}. 相似度分数: {result['score']:.4f}")
        info(f"     文档: {result['text'][:100]}...")
        info("")

    # 6. 加载文件存储的用户反馈数据
    info("---6--->正在加载用户反馈数据...")
    feedback_data_array = load_feedback_data()

    new_top_chunks = optimize_top_chunks_score(
        query, top_chunks, feedback_data_array)

    info("6.优化后的搜索结果:")
    for i, result in enumerate(new_top_chunks):
        info(f"{i+1}. 相似度分数: {result['score']:.4f}")
        info(f"   文档: {result['text'][:100]}...")
        info("")

    system_prompt = """
    你是一个AI助手，请严格根据以下信息回答问题。如果信息中没有答案，请回答“我不知道”。"""

    user_prompt = "\n".join(
        [f"上下文内容 {i + 1} :\n{result['text']}\n========\n"
         for i, result in enumerate(new_top_chunks)])

    user_prompt = f"{user_prompt}\n\n Question: {query}"

    # 7. 调用LLM模型，生成回答
    result = query_llm(system_prompt, user_prompt)
    info(f"---7--->final result: {result}")

    # 第5步：收集用户反馈以改进未来的表现
    print("\n=== 您是否愿意对这个响应提供反馈？ ===")

    print("评分质量（1-5，5表示最高质量）：")
    quality = input()

    expanded_questions = expand_question_with_llm(query, result)

    # 第8步：将反馈格式化为结构化数据
    feedback = {
        "query": query,
        "response": result,
        "quality": int(quality),
        "related_chunks": new_top_chunks,
        "expanded_questions": expanded_questions
    }

    info(f"8.feedback: 将原始问题转化{query}成了{expanded_questions}")

    # 第7步：持久化反馈以实现系统的持续学习
    store_feedback(feedback)
    print("反馈已记录。感谢您的参与！")
