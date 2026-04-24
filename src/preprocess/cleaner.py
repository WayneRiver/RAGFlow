from pathlib import Path
import sys

# 将项目根目录添加到 Python 路径，以便导入 config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def clean_logs(logs: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    清洗日志数据
    
    步骤：
    1. 去除空消息
    2. 统一日志级别（WARN -> WARNING, FATAL -> CRITICAL）
    3. 去除重复日志（基于时间+消息判断）
    4. 清理多余空白字符
    
    参数:
        logs: 解析后的日志列表，每个元素包含 time、level、message
    
    返回:
        清洗后的日志列表
    """

    # 定义日志级别映射：将非标准级别统一为标准级别
    # WARN -> WARNING, FATAL -> CRITICAL
    LEVEL_MAPPING = {
        "WARN": "WARNING",
        "FATAL": "CRITICAL",
    }

    # 用于去重，记录已见过的 (时间, 消息) 组合
    seen_messages = set()
    cleaned = []

    for log in logs:
        # 跳过空消息
        message = log.get("message", "")
        if not message or not message.strip():
            continue

        # 统一日志级别
        level = log.get("level", "INFO").upper()
        level = LEVEL_MAPPING.get(level, level)

        # 过滤无效级别
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid_levels:
            level = "INFO"

        # 清理消息中的多余空白
        message = " ".join(message.split())

        # 去重
        log_key = (log.get("time", ""), message)
        if log_key in seen_messages:
            continue
        seen_messages.add(log_key)
        cleaned.append({
            "time": log.get("time", ""),
            "level": level,
            "message": message,
        })
    
    return cleaned

def filter_by_level(logs: list[dict[str, str]], min_level: str = "INFO") -> list[dict[str, str]]:
    """
    根据日志级别过滤日志
    
    只保留 >= min_level 的日志
    级别从低到高: DEBUG < INFO < WARNING < ERROR < CRITICAL
    
    参数:
        logs: 日志列表
        min_level: 最低级别，默认 INFO
    
    返回:
        过滤后的日志列表
    """

    # 定义级别顺序（数字越大级别越高）
    LEVEL_ORDER = {
        "DEBUG": 0,
        "INFO": 1,
        "WARNING": 2,
        "ERROR": 3,
        "CRITICAL": 4,
    }

    min_level_value = LEVEL_ORDER.get(min_level.upper(), 1)

    filtered = []
    for log in logs:
        level = log.get("level", "INFO").upper()
        level_value = LEVEL_ORDER.get(level, 1)
        
        if level_value >= min_level_value:
            filtered.append(log)

    return filtered

def get_log_stats(logs: list[dict[str, str]]) -> dict:
    """
    获取日志统计信息
    
    参数:
        logs: 日志列表
    
    返回:
        统计信息字典，包含:
        - total: 总数
        - by_level: 各级别数量
        - unique_messages: 唯一消息数
    """

    if not logs:
        return {
            "total": 0,
            "by_level": {},
            "unique_messages": 0,
        }
    
    # 统计各级别数量
    by_level = {}
    for log in logs:
        level = log.get("level", "INFO")
        by_level[level] = by_level.get(level, 0) + 1
    
    # 统计唯一消息数
    unique_messages = len(set(log["message"] for log in logs))

    return {
        "total": len(logs),
        "by_level": by_level,
        "unique_messages": unique_messages,
    }

# ===== 测试代码 =====
if __name__ == "__main__":
    from parser import parse_log_file
    
    # 解析日志
    from config import RAW_DATA_DIR
    log_file = RAW_DATA_DIR / "test.log"
    if log_file.exists():
        logs = parse_log_file(log_file)
        print(f"解析得到 {len(logs)} 条日志")
        
        # 清洗日志
        cleaned = clean_logs(logs)
        print(f"清洗后得到 {len(cleaned)} 条日志")
        print("\n清洗后的日志:")
        for log in cleaned:
            print(f"  [{log['level']}] {log['time']} - {log['message']}")
            
        # 打印统计
        stats = get_log_stats(cleaned)
        print(f"\n统计信息: {stats}")

        # ===== 测试 filter_by_level =====
        print("\n===== 测试 filter_by_level =====")
        
        # 测试1: 只保留 WARNING 及以上
        result = filter_by_level(cleaned, min_level="WARNING")
        print(f"只保留 WARNING 及以上: {len(result)} 条")
        for log in result:
            print(f"  [{log['level']}] {log['time']} - {log['message']}")

        # 测试2: 只保留 ERROR 及以上
        result = filter_by_level(cleaned, min_level="ERROR")
        print(f"\n只保留 ERROR 及以上: {len(result)} 条")
        for log in result:
            print(f"  [{log['level']}] {log['time']} - {log['message']}")
    else:
        print(f"测试文件不存在: {log_file}")