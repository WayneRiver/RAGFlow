from pathlib import Path
import sys

# 将项目根目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def chunk_logs(logs: list[dict[str, str]], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    将日志消息分块（按字符数）
    
    逻辑：
    - 如果消息 <= chunk_size，直接作为一个块
    - 如果消息 > chunk_size，切割成多个块，块之间有重叠（保持上下文）
    
    参数:
        logs: 清洗后的日志列表
        chunk_size: 每个块的最大字符数，默认 500
        overlap: 块之间的重叠字符数，默认 50
    
    返回:
        分块列表，每个元素包含:
        - text: 块文本内容
        - metadata: 元数据（时间、级别、索引等）
    """

    chunks = []

    for i, log in enumerate(logs):
        message = log.get("message", "")

        if len(message) <= chunk_size or chunk_size <= overlap:
            chunks.append({
                "text": message,
                "metadata": {
                    "time": log.get("time", ""),
                    "level": log.get("level", "INFO"),
                    "log_index": i,
                    "chunk_type": "single"
                }
            })
        else:
            # 分割消息
            start = 0
            chunk_num = 0
            actual_step = chunk_size - overlap
            total_chunks = (len(message) + actual_step - 1) // actual_step

            while start < len(message):
                # 取出当前块
                end = start + chunk_size
                chunk_text = message[start:end]

                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "time": log.get("time", ""),
                        "level": log.get("level", "INFO"),
                        "log_index": i,
                        "chunk_type": "split",
                        "chunk_num": chunk_num,
                        "total_chunks": total_chunks
                    }
                })
                chunk_num += 1
                start += actual_step

    return chunks

def chunk_by_time(logs: list[dict[str, str]], time_window_minutes: int = 5) -> list[dict]:
    """
    按时间窗口分块
    
    将时间相近的日志聚合到一个块中，适合分析连续事件
    例如：5分钟内的所有日志合并成一个块
    
    参数:
        logs: 日志列表
        time_window_minutes: 时间窗口大小（分钟），默认 5 分钟
    
    返回:
        分块列表，每个元素包含 text、metadata
    """
    from datetime import datetime, timedelta

    if not logs:
        return []

    # ===== 先按时间排序，确保日志顺序正确 =====
    sorted_logs = []
    for log in logs:
        time_str = log.get("time", "")
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
            try:
                log_time = datetime.strptime(time_str, fmt)
                # 无法解析时间，跳过该日志
                if log_time is None:
                    continue
                sorted_logs.append((log_time, log))
                break
            except:
                continue
    
    # 按时间升序排序
    sorted_logs.sort(key=lambda x: x[0])

    chunks = []
    current_chunk = []        # 当前时间窗口内的日志
    current_start_time = None # 当前时间窗口的开始时间

    for log_time, log in sorted_logs:
        # ===== 初始化第一个时间窗口 =====
        if current_start_time is None:
            current_start_time = log_time
            current_chunk = [log]
            continue

        # ===== 判断是否在当前时间窗口内 =====
        time_diff = (log_time - current_start_time).total_seconds()
        if time_diff <= time_window_minutes * 60:
            current_chunk.append(log)
        else:
            # 超过时间窗口，保存当前块，开始新块
            chunk_text = "\n".join([l["message"] for l in current_chunk])
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "time": current_chunk[0].get("time", ""),
                    "level": current_chunk[-1].get("level", "INFO"),
                    "chunk_type": "time_window",
                    "log_count": len(current_chunk)
                }
            })

            # 开始新的时间窗口
            current_start_time = log_time
            current_chunk = [log]

    # ===== 处理最后一个块 =====
    if current_chunk:
        chunk_text = "\n".join([l["message"] for l in current_chunk])
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "time": current_chunk[0].get("time", ""),
                "level": current_chunk[-1].get("level", "INFO"),
                "chunk_type": "time_window",
                "log_count": len(current_chunk)
            }
        })
    
    return chunks


# ===== 测试代码 =====
if __name__ == "__main__":
    # 模拟测试数据
    test_logs = [
        {"time": "2024-01-01 10:30:45", "level": "ERROR", "message": "数据库连接失败"},
        {"time": "2024-01-01 10:30:46", "level": "INFO", "message": "服务启动成功"},
        {"time": "2024-01-01 10:30:47", "level": "WARNING", "message": "内存使用率超过75%，请及时处理" + "x" * 500},  # 长消息
    ]
    
    print("===== 测试 chunk_logs =====")
    
    # 测试1: 默认参数
    chunks = chunk_logs(test_logs)
    print(f"输入: {len(test_logs)} 条日志")
    print(f"输出: {len(chunks)} 个块\n")
    
    # 打印每个块
    for i, chunk in enumerate(chunks):
        print(f"--- 块 {i+1} ---")
        print(f"内容: {chunk['text'][:50]}...")
        print(f"字符数: {len(chunk['text'])}")
        print(f"元数据: {chunk['metadata']}")
        print()
    
    # ===== 测试 chunk_by_time =====
    print("\n===== 测试 chunk_by_time =====")

    test_logs_time = [
        {"time": "2024-01-01 10:00:00", "level": "INFO", "message": "服务启动"},
        {"time": "2024-01-01 10:02:00", "level": "INFO", "message": "检查配置"},
        {"time": "2024-01-01 10:04:00", "level": "ERROR", "message": "连接失败"},
        {"time": "2024-01-01 10:10:00", "level": "ERROR", "message": "重试连接"},
        {"time": "2024-01-01 10:12:00", "level": "WARNING", "message": "告警"},
    ]

    chunks = chunk_by_time(test_logs_time, time_window_minutes=5)
    print(f"输入: {len(test_logs_time)} 条日志")
    print(f"输出: {len(chunks)} 个时间块\n")

    for i, chunk in enumerate(chunks):
        print(f"--- 块 {i+1} ---")
        print(f"内容: {chunk['text']}")
        print(f"日志数: {chunk['metadata']['log_count']}")
        print()
