import datetime


def info(message):
    """打印带时间戳的信息消息"""
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{current_time} : {message}")


def error(message):
    """打印带时间戳的错误消息"""
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{current_time} - ERROR : - {message}")
