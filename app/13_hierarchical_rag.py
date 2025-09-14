from tqdm import tqdm
from utils.common_utils import chunk_by_chapters, chunk_text_by_length
from utils.embedding_model import EmbeddingModel
from utils.file_utils import extract_text_from_markdown
from utils.llm_utils import query_llm
from utils.similarity_utils import similar_search
from utils.logger_utils import info

# 0. 构建全局向量模型
embedding_model = EmbeddingModel()


def generate_page_summary(page_text):
    """
    生成页面的简洁摘要。

    Args:
        page_text (str): 页面的文本内容

    Returns:
        str: 生成的摘要
    """
    # 定义系统提示，指导摘要模型如何生成摘要
    system_prompt = """你是一个专业的摘要生成系统。
    请对提供的文本创建一个详细的摘要。
    重点捕捉主要内容、关键信息和重要事实。
    你的摘要应足够全面，能够让人理解该页面包含的内容，
    但要比原文更简洁。"""

    # 如果输入文本超过最大令牌限制，则截断
    max_tokens = 6000
    truncated_text = page_text[:max_tokens] if len(
        page_text) > max_tokens else page_text

    user_prompt = f"请总结以下文本:\n\n{truncated_text}"

    # 向OpenAI API发出请求以生成摘要
    return query_llm(system_prompt, user_prompt, temperature=0.3)


if __name__ == "__main__":
    query = "孙悟空的兵器是什么？"

    # 1. 提取文本
    info("---1--->正在提取西游记文本...")
    extract_text = extract_text_from_markdown()
    # print(f"文本长度: {len(extract_text)} 字符")

    # 2. 分割文本
    info("---2--->正在分割文本...")
    chapters = chunk_by_chapters(extract_text)

    # 3. 生成章节摘要
    info("---3--->正在生成章节摘要...")
    chapters_summaries = []
    for i in tqdm(range(len(chapters)), "正在生成章节摘要"):
        chapters_summaries.append(
            generate_page_summary(chapters[i]['content']))

    # 4. 为每个章节内容创建详细块
    info("---4--->正在为每个章节内容创建详细块...")
    content_chunks = []
    for i in tqdm(range(len(chapters)), "正在为每个章节内容创建详细块"):
        details_chunks_text_array = chunk_text_by_length(
            chapters[i]['content'], 300, 100)
        chapters[i]['details_chunks'] = []
        for j, single_detail_chunk in enumerate(details_chunks_text_array):
            chapters[i]['details_chunks'].append(
                {"detail_index": j, "detail_content": single_detail_chunk})

    # 打印前3个章节的信息
    for i, chapter in enumerate(chapters):
        info(f"章节 {chapter['index']}: {chapter['title']}")
        info(f"内容长度: {len(chapter['content'])} 字符")
        info(f"摘要: {chapters_summaries[i]}")
        info(f"详细块数量: {len(chapter['details_chunks'])}")
        info("-" * 50)

    # 4. 为每个章节中的摘要和内容都分别创建向量
    chapters_embeddings = []
    info("---4--->正在为每个章节中的摘要和内容都分别创建向量...")
    for i in tqdm(range(len(chapters)), "为每个章节中的摘要和内容都分别创建向量"):
        chapters_embeddings.append(embedding_model.create_embeddings(
            chapters_summaries[i]))
        details_chunks = chapters[i]['details_chunks']
        for j, chunk in enumerate(details_chunks):
            chapters[i]['details_chunks'][j]['detail_embedding'] = embedding_model.create_embeddings(
                details_chunks[j]["detail_content"])

    # 5. 构建问题向量
    info("---5--->正在构建问题向量...")
    query_embeddings = embedding_model.create_embeddings([query])

    # 6. 先拿用户问题和章节摘要进行相似度检索
    info("---6--->先拿用户问题和章节摘要进行相似度检索...")
    top_summaries = similar_search(
        chapters_summaries, chapters_embeddings, query_embeddings, 5)
    info(f"章节摘要相似度检索结果: {top_summaries}")

    # 7. 将这些摘要所属的章节内容中的所有片段都集合在一起
    info("---7--->将这些摘要所属的章节内容中的所有片段都集合在一起...")
    detail_result_chunks_content = []
    detail_result_chunks_embeddings = []
    for i, summary in enumerate(top_summaries):
        single_chapter_chunks = chapters[summary['index']]['details_chunks']
        for j, detail_chunk in enumerate(single_chapter_chunks):
            detail_result_chunks_content.append(detail_chunk['detail_content'])
            detail_result_chunks_embeddings.append(
                detail_chunk['detail_embedding'])

    # 8. 再拿用户问题和这些片段进行相似度检索
    info("---8--->再拿用户问题和这些片段进行相似度检索...")
    top_detail_result_chunks = similar_search(
        detail_result_chunks_content, detail_result_chunks_embeddings, query_embeddings, 5)
    info(f"章节内容相似度检索结果: {top_detail_result_chunks}")

    info("搜索结果:")
    for i, result in enumerate(top_detail_result_chunks):
        info(f"{i+1}. 相似度分数: {result['score']:.4f}")
        info(f"   文档: {result['text'][:100]}...")
        info("")

    system_prompt = """
    你是一个AI助手，请严格根据以下信息回答问题。如果信息中没有答案，请回答“我不知道”。"""

    user_prompt = "\n".join(
        [f"上下文内容 {i + 1} :\n{result['text']}\n========\n"
         for i, result in enumerate(top_detail_result_chunks)])

    user_prompt = f"{user_prompt}\n\n Question: {query}"

    # 7. 调用LLM模型，生成回答
    result = query_llm(system_prompt, user_prompt)
    info(f"---7--->final result: {result}")
