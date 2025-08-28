

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
