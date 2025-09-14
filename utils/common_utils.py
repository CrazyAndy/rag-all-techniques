

import re
from utils.logger_utils import info


def validate_chunk_text_params(text, single_chunk_size, overlap):
    if not text:
        return []
    if single_chunk_size <= 0:
        raise ValueError("single_chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= single_chunk_size:
        raise ValueError("overlap must be less than single_chunk_size")


def chunk_text_by_length(text, single_chunk_size, overlap):
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
    chunks = []
    for i in range(0, len(text), single_chunk_size - overlap):
        chunks.append(text[i:i + single_chunk_size])
    return chunks


def chunk_by_chapters(text):
    """
    根据章节标题切分文档
    
    Args:
        text (str): 完整的文档文本
        
    Returns:
        list: 包含章节信息的列表，每个元素包含：
            - index: 章节索引
            - title: 章节标题
            - content: 章节内容
    """
    chapters = []
    
    # 使用正则表达式匹配章节标题
    # 匹配格式：# 第X回　标题内容_西游记白话文小说
    chapter_pattern = r'^# 第(\d+)回　(.+?)_西游记白话文小说$'
    
    # 按行分割文本
    lines = text.split('\n')
    current_chapter = None
    current_content = []
    
    for line in lines:
        # 检查是否是章节标题
        match = re.match(chapter_pattern, line)
        if match:
            # 如果之前有章节，先保存它
            if current_chapter is not None:
                chapters.append({
                    'index': current_chapter['index'],
                    'title': current_chapter['title'],
                    'content': '\n'.join(current_content).strip()
                })
            
            # 开始新章节
            chapter_index = int(match.group(1))
            chapter_title = match.group(2)
            current_chapter = {
                'index': chapter_index,
                'title': chapter_title
            }
            current_content = []
        else:
            # 如果不是章节标题，添加到当前章节内容
            if current_chapter is not None:
                current_content.append(line)
    
    # 保存最后一个章节
    if current_chapter is not None:
        chapters.append({
            'index': current_chapter['index'],
            'title': current_chapter['title'],
            'content': '\n'.join(current_content).strip()
        })
    
    info(f"成功切分出 {len(chapters)} 个章节")
    return chapters

def chunk_by_chapters_with_sections(text, section_size=500):
    """
    根据章节标题切分文档，并进一步将每个章节按大小切分
    
    Args:
        text (str): 完整的文档文本
        section_size (int): 每个section的最大字符数
        
    Returns:
        list: 包含章节信息的列表，每个元素包含：
            - index: 章节索引
            - title: 章节标题
            - content: 章节内容
            - section_index: 在章节内的段落索引（如果章节被分割）
    """
    chapters = chunk_by_chapters(text)
    sections = []
    
    for chapter in chapters:
        content = chapter['content']
        
        # 如果章节内容太长，按段落分割
        if len(content) > section_size:
            # 按段落分割（双换行符）
            paragraphs = content.split('\n\n')
            current_section = []
            current_length = 0
            section_index = 0
            
            for paragraph in paragraphs:
                paragraph_length = len(paragraph)
                
                # 如果当前段落加上现有内容超过限制，保存当前section
                if current_length + paragraph_length > section_size and current_section:
                    sections.append({
                        'index': chapter['index'],
                        'title': chapter['title'],
                        'content': '\n\n'.join(current_section),
                        'section_index': section_index,
                        'full_title': f"第{chapter['index']}回 {chapter['title']} (第{section_index + 1}段)"
                    })
                    current_section = []
                    current_length = 0
                    section_index += 1
                
                current_section.append(paragraph)
                current_length += paragraph_length
            
            # 保存最后一个section
            if current_section:
                sections.append({
                    'index': chapter['index'],
                    'title': chapter['title'],
                    'content': '\n\n'.join(current_section),
                    'section_index': section_index,
                    'full_title': f"第{chapter['index']}回 {chapter['title']} (第{section_index + 1}段)"
                })
        else:
            # 章节内容不长，直接保存
            sections.append({
                'index': chapter['index'],
                'title': chapter['title'],
                'content': content,
                'section_index': 0,
                'full_title': f"第{chapter['index']}回 {chapter['title']}"
            })
    
    info(f"成功切分出 {len(sections)} 个章节段落")
    return sections


def test_chunk_by_chapters():
    """
    测试章节切分功能
    """
    from utils.file_utils import extract_text_from_markdown
    
    # 读取文档
    text = extract_text_from_markdown()
    
    # 按章节切分
    chapters = chunk_by_chapters(text)
    
    # 打印前3个章节的信息
    for i, chapter in enumerate(chapters[:3]):
        print(f"章节 {chapter['index']}: {chapter['title']}")
        print(f"内容长度: {len(chapter['content'])} 字符")
        print(f"内容预览: {chapter['content'][:100]}...")
        print("-" * 50)
    
    # 按章节和段落切分
    sections = chunk_by_chapters_with_sections(text, section_size=1000)
    
    # 打印前3个段落的信息
    for i, section in enumerate(sections[:3]):
        print(f"段落 {section['full_title']}")
        print(f"内容长度: {len(section['content'])} 字符")
        print(f"内容预览: {section['content'][:100]}...")
        print("-" * 50)


if __name__ == "__main__":
    test_chunk_by_chapters()
