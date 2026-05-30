#!/usr/bin/env python3
"""
验证器 - 检查GIF是否符合Slack的要求。

这些验证器有助于确保您的GIF符合Slack的大小和尺寸限制。
"""

from pathlib import Path


def check_slack_size(gif_path: str | Path, is_emoji: bool = True) -> tuple[bool, dict]:
    """
    检查GIF是否符合Slack的大小限制。

    参数:
        gif_path: GIF文件路径
        is_emoji: True表示表情GIF（64KB限制），False表示消息GIF（2MB限制）

    返回:
        (passes: bool, info: dict) 元组，包含详情的字典
    """
    gif_path = Path(gif_path)

    if not gif_path.exists():
        return False, {"error": f"File not found: {gif_path}"}

    size_bytes = gif_path.stat().st_size
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024

    limit_kb = 64 if is_emoji else 2048
    limit_mb = limit_kb / 1024

    passes = size_kb <= limit_kb

    info = {
        "size_bytes": size_bytes,
        "size_kb": size_kb,
        "size_mb": size_mb,
        "limit_kb": limit_kb,
        "limit_mb": limit_mb,
        "passes": passes,
        "type": "emoji" if is_emoji else "message",
    }

    # 打印反馈信息
    if passes:
        print(f"✓ {size_kb:.1f} KB - 在 {limit_kb} KB 限制内")
    else:
        print(f"✗ {size_kb:.1f} KB - 超过 {limit_kb} KB 限制")
        overage_kb = size_kb - limit_kb
        overage_percent = (overage_kb / limit_kb) * 100
        print(f"  超出了: {overage_kb:.1f} KB ({overage_percent:.1f}%)")
        print("  尝试: 更少的帧数、更少的颜色或更简单的设计")

    return passes, info


def validate_dimensions(width: int, height: int, is_emoji: bool = True) -> tuple[bool, dict]:
    """
    检查尺寸是否适合Slack。

    参数:
        width: 帧宽度（像素）
        height: 帧高度（像素）
        is_emoji: True表示表情GIF，False表示消息GIF

    返回:
        (passes: bool, info: dict) 元组，包含详情的字典
    """
    info = {"width": width, "height": height, "is_square": width == height, "type": "emoji" if is_emoji else "message"}

    if is_emoji:
        # 表情GIF应该是128x128
        optimal = width == height == 128
        acceptable = width == height and 64 <= width <= 128

        info["optimal"] = optimal
        info["acceptable"] = acceptable

        if optimal:
            print(f"✓ {width}x{height} - 表情的最佳尺寸")
            passes = True
        elif acceptable:
            print(f"⚠ {width}x{height} - 可接受但128x128是最佳尺寸")
            passes = True
        else:
            print(f"✗ {width}x{height} - 表情应该是正方形，推荐128x128")
            passes = False
    else:
        # 消息GIF应该是近似正方形且尺寸合理
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float("inf")
        reasonable_size = 320 <= min(width, height) <= 640

        info["aspect_ratio"] = aspect_ratio
        info["reasonable_size"] = reasonable_size

        # 检查是否大致为正方形（在2:1比例内）
        is_square_ish = aspect_ratio <= 2.0

        if is_square_ish and reasonable_size:
            print(f"✓ {width}x{height} - 适合消息GIF")
            passes = True
        elif is_square_ish:
            print(f"⚠ {width}x{height} - 近似正方形但尺寸不常见")
            passes = True
        elif reasonable_size:
            print(f"⚠ {width}x{height} - 尺寸合适但不是近似正方形")
            passes = True
        else:
            print(f"✗ {width}x{height} - Slack尺寸不寻常")
            passes = False

    return passes, info


def validate_gif(gif_path: str | Path, is_emoji: bool = True) -> tuple[bool, dict]:
    """
    对GIF文件运行所有验证。

    参数:
        gif_path: GIF文件路径
        is_emoji: True表示表情GIF，False表示消息GIF

    返回:
        (all_pass: bool, results: dict) 元组
    """
    from PIL import Image

    gif_path = Path(gif_path)

    if not gif_path.exists():
        return False, {"error": f"File not found: {gif_path}"}

    print(f"\n验证 {gif_path.name} 作为 {'emoji' if is_emoji else 'message'} GIF:")
    print("=" * 60)

    # 检查文件大小
    size_pass, size_info = check_slack_size(gif_path, is_emoji)

    # 检查尺寸
    try:
        with Image.open(gif_path) as img:
            width, height = img.size
            dim_pass, dim_info = validate_dimensions(width, height, is_emoji)

            # 统计帧数
            frame_count = 0
            try:
                while True:
                    img.seek(frame_count)
                    frame_count += 1
            except EOFError:
                pass

            # 获取持续时间（如果可用）
            try:
                duration_ms = img.info.get("duration", 100)
                total_duration = (duration_ms * frame_count) / 1000
                fps = frame_count / total_duration if total_duration > 0 else 0
            except:
                duration_ms = None
                total_duration = None
                fps = None

    except Exception as e:
        return False, {"error": f"Failed to read GIF: {e}"}

    print(f"\n帧数: {frame_count}")
    if total_duration:
        print(f"持续时间: {total_duration:.1f}s @ {fps:.1f} fps")

    all_pass = size_pass and dim_pass

    results = {
        "file": str(gif_path),
        "passes": all_pass,
        "size": size_info,
        "dimensions": dim_info,
        "frame_count": frame_count,
        "duration_seconds": total_duration,
        "fps": fps,
    }

    print("=" * 60)
    if all_pass:
        print("✓ 所有验证通过！")
    else:
        print("✗ 部分验证失败")
    print()

    return all_pass, results


def get_optimization_suggestions(results: dict) -> list[str]:
    """
    根据验证结果获取优化GIF的建议。

    参数:
        results: 来自validate_gif()的结果字典

    返回:
        建议字符串列表
    """
    suggestions = []

    if not results.get("passes", False):
        size_info = results.get("size", {})
        dim_info = results.get("dimensions", {})

        # 大小建议
        if not size_info.get("passes", True):
            overage = size_info["size_kb"] - size_info["limit_kb"]
            if size_info["type"] == "emoji":
                suggestions.append(f"减少 {overage:.1f} KB 文件大小:")
                suggestions.append("  - 限制在10-12帧")
                suggestions.append("  - 使用最多32-40种颜色")
                suggestions.append("  - 移除渐变（纯色压缩更好）")
                suggestions.append("  - 简化设计")
            else:
                suggestions.append(f"减少 {overage:.1f} KB 文件大小:")
                suggestions.append("  - 减少帧数或FPS")
                suggestions.append("  - 使用更少的颜色（128 → 64）")
                suggestions.append("  - 减小尺寸")

        # 尺寸建议
        if not dim_info.get("optimal", True) and dim_info.get("type") == "emoji":
            suggestions.append("最佳表情GIF:")
            suggestions.append("  - 使用128x128尺寸")
            suggestions.append("  - 确保宽高比为正方形")

    return suggestions


# 快速检查的便捷函数
def is_slack_ready(gif_path: str | Path, is_emoji: bool = True, verbose: bool = True) -> bool:
    """
    快速检查GIF是否已准备好用于Slack。

    参数:
        gif_path: GIF文件路径
        is_emoji: True表示表情GIF，False表示消息GIF
        verbose: 打印详细反馈

    返回:
        如果已准备好则返回True，否则返回False
    """
    if verbose:
        passes, results = validate_gif(gif_path, is_emoji)
        if not passes:
            suggestions = get_optimization_suggestions(results)
            if suggestions:
                print("\n建议:")
                for suggestion in suggestions:
                    print(suggestion)
        return passes
    else:
        size_pass, _ = check_slack_size(gif_path, is_emoji)
        return size_pass
