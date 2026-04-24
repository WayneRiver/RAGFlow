import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def parse_log_file(file_path: Path) -> list[dict[str, str]]:
    """
    解析日志文件，提取时间、级别、消息
    
    参数:
        file_path: 日志文件路径
    
    返回:
        列表，每个元素是一个字典，包含 time、level、message
    """

    # 日志正则表达式：支持多种常见格式
    # 格式1: 2024-01-01 10:30:45 ERROR 数据库连接失败
    # 格式2: 2024/01/01 10:30:45 [ERROR] 数据库连接失败
    pattern = (r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})\s+'
          r'(?:\[)?(DEBUG|INFO|WARNING|WARN|ERROR|FATAL|CRITICAL)(?:\])?\s+'
          r'(.+)')

    results = []

    with open(str(file_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            #用正则匹配
            match = re.match(pattern, line)
            if match:
                results.append({
                    "time": match.group(1),
                    "level": match.group(2),
                    "message": match.group(3)
                })

    return results

def parse_log_lines(lines: list[str]) -> list[dict[str, str]]:
    """
    解析多行日志字符串
    
    参数:
        lines: 日志行列表
    
    返回:
        解析后的字典列表
    """
    pattern = (r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})\s+'
          r'(?:\[)?(DEBUG|INFO|WARNING|WARN|ERROR|FATAL|CRITICAL)(?:\])?\s+'
          r'(.+)')
    
    results = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            results.append({
                "time": match.group(1),
                "level": match.group(2),
                "message": match.group(3)
            })
    return results

if __name__ == "__main__":
    # 测试代码
    print("测试 parse_log_lines: ")
    test_log = [
        "2024-01-01 10:30:45 ERROR 数据库连接失败",
        "2024-01-01 10:30:46 INFO 服务启动成功",
        "2024-01-01 10:30:47 [WARN] 内存使用率超过80%",
    ]
    result = parse_log_lines(test_log)
    print("解析结果：")
    for item in result:
        print(item)

    print("-" * 50)
    print("测试 parse_log_file: ")
    from config import RAW_DATA_DIR
    test_file = RAW_DATA_DIR / "test.log"
    if test_file.exists():
        result = parse_log_file(str(test_file))
        print("解析结果：")
        for item in result:
            print(item)
    else:
        print(f"文件不存在: {test_file}")
    