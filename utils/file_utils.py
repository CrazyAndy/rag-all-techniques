from pathlib import Path

def extract_text_from_markdown():
    """
    读取 data/xiyouji.md 文件的内容
    
    Returns:
        str: 西游记文件的内容，如果文件不存在则返回空字符串
    """
    # 获取项目根目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # 构建文件路径
    file_path = project_root / "data" / "xiyouji.md"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return ""
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return ""


