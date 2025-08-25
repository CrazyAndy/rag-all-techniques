

class ProgressBar:
    """
    进度条类，用于显示处理进度
    
    使用方法:
    progress = ProgressBar(total_items=100, description="处理中")
    for i in range(100):
        # 处理某个任务
        progress.update((i + 1) / 100 * 100)
    progress.finish()
    """
    
    def __init__(self, total_items=None, description="处理进度", bar_length=20):
        """
        初始化进度条
        
        Args:
            total_items: 总项目数（可选）
            description: 进度条描述
            bar_length: 进度条长度（字符数）
        """
        self.description = description
        self.bar_length = bar_length
        self.total_items = total_items
        self.current_percentage = 0
        
        # 显示初始进度条
        self._display_progress(0)
    
    def update(self, percentage):
        """
        更新进度条
        
        Args:
            percentage: 百分比值 (0-100)
        """
        if percentage < 0:
            percentage = 0
        elif percentage > 100:
            percentage = 100
            
        self.current_percentage = percentage
        self._display_progress(percentage)
    
    def update_by_count(self, current_count):
        """
        通过当前计数更新进度条
        
        Args:
            current_count: 当前处理的项目数
        """
        if self.total_items is None:
            raise ValueError("必须设置 total_items 才能使用 update_by_count 方法")
        
        percentage = (current_count / self.total_items) * 100
        self.update(percentage)
    
    def finish(self):
        """
        完成进度条，显示100%
        """
        self.update(100)
        print("\n")  # 换行
    
    def _display_progress(self, percentage):
        """
        内部方法：显示进度条
        
        Args:
            percentage: 百分比值
        """
        filled_length = int(self.bar_length * percentage / 100)
        bar = "█" * filled_length + "░" * (self.bar_length - filled_length)
        
        print(f"\r{self.description}: [{bar}] {percentage:.1f}%", end="", flush=True)


# 便捷函数
def create_progress_bar(total_items=None, description="处理进度", bar_length=20):
    """
    创建进度条实例的便捷函数
    
    Args:
        total_items: 总项目数（可选）
        description: 进度条描述
        bar_length: 进度条长度
        
    Returns:
        ProgressBar: 进度条实例
    """
    return ProgressBar(total_items, description, bar_length)


