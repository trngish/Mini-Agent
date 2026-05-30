#!/usr/bin/env python3
"""将文本替换应用到 PowerPoint 演示文稿。

用法：
    python replace.py <input.pptx> <replacements.json> <output.pptx>

replacements JSON 应具有 inventory.py 输出的结构。
除非为该形状指定了 "paragraphs"，否则 inventory.py 识别的所有文本形状的文本都将被清除。
"""

import json
import sys
from pathlib import Path
from typing import Any

from inventory import InventoryData, extract_text_inventory
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt


def clear_paragraph_bullets(paragraph):
    """清除段落中的项目符号格式。"""
    pPr = paragraph._element.get_or_add_pPr()

    # 移除现有的项目符号元素
    for child in list(pPr):
        if (
            child.tag.endswith("buChar")
            or child.tag.endswith("buNone")
            or child.tag.endswith("buAutoNum")
            or child.tag.endswith("buFont")
        ):
            pPr.remove(child)

    return pPr


def apply_paragraph_properties(paragraph, para_data: dict[str, Any]):
    """将格式属性应用到段落。"""
    # 获取文本，但暂不直接设置到段落
    text = para_data.get("text", "")

    # 获取或创建段落属性
    pPr = clear_paragraph_bullets(paragraph)

    # 处理项目符号格式
    if para_data.get("bullet", False):
        level = para_data.get("level", 0)
        paragraph.level = level

        # 计算与字体成比例的缩进
        font_size = para_data.get("font_size", 18.0)
        level_indent_emu = int((font_size * (1.6 + level * 1.6)) * 12700)
        hanging_indent_emu = int(-font_size * 0.8 * 12700)

        # 设置缩进
        pPr.attrib["marL"] = str(level_indent_emu)
        pPr.attrib["indent"] = str(hanging_indent_emu)

        # 添加项目符号字符
        buChar = OxmlElement("a:buChar")
        buChar.set("char", "•")
        pPr.append(buChar)

        # 如果未指定，默认左对齐项目符号
        if "alignment" not in para_data:
            paragraph.alignment = PP_ALIGN.LEFT
    else:
        # 为非项目符号文本移除缩进
        pPr.attrib["marL"] = "0"
        pPr.attrib["indent"] = "0"

        # 添加 buNone 元素
        buNone = OxmlElement("a:buNone")
        pPr.insert(0, buNone)

    # 应用对齐方式
    if "alignment" in para_data:
        alignment_map = {
            "LEFT": PP_ALIGN.LEFT,
            "CENTER": PP_ALIGN.CENTER,
            "RIGHT": PP_ALIGN.RIGHT,
            "JUSTIFY": PP_ALIGN.JUSTIFY,
        }
        if para_data["alignment"] in alignment_map:
            paragraph.alignment = alignment_map[para_data["alignment"]]

    # 应用间距
    if "space_before" in para_data:
        paragraph.space_before = Pt(para_data["space_before"])
    if "space_after" in para_data:
        paragraph.space_after = Pt(para_data["space_after"])
    if "line_spacing" in para_data:
        paragraph.line_spacing = Pt(para_data["line_spacing"])

    # 应用 run 级别的格式
    if not paragraph.runs:
        run = paragraph.add_run()
        run.text = text
    else:
        run = paragraph.runs[0]
        run.text = text

    # 应用字体属性
    apply_font_properties(run, para_data)


def apply_font_properties(run, para_data: dict[str, Any]):
    """将字体属性应用到文本 run。"""
    if "bold" in para_data:
        run.font.bold = para_data["bold"]
    if "italic" in para_data:
        run.font.italic = para_data["italic"]
    if "underline" in para_data:
        run.font.underline = para_data["underline"]
    if "font_size" in para_data:
        run.font.size = Pt(para_data["font_size"])
    if "font_name" in para_data:
        run.font.name = para_data["font_name"]

    # 应用颜色 - 优先使用 RGB，回退到 theme_color
    if "color" in para_data:
        color_hex = para_data["color"].lstrip("#")
        if len(color_hex) == 6:
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            run.font.color.rgb = RGBColor(r, g, b)
    elif "theme_color" in para_data:
        # 通过名称获取主题颜色（例如 "DARK_1"、"ACCENT_1"）
        theme_name = para_data["theme_color"]
        try:
            run.font.color.theme_color = getattr(MSO_THEME_COLOR, theme_name)
        except AttributeError:
            print(f"  警告：未知的主题颜色名称 '{theme_name}'")


def detect_frame_overflow(inventory: InventoryData) -> dict[str, dict[str, float]]:
    """检测形状中的文本溢出（文本超出形状边界）。

    返回 slide_key -> shape_key -> overflow_inches 的字典。
    仅包括有文本溢出的形状。
    """
    overflow_map = {}

    for slide_key, shapes_dict in inventory.items():
        for shape_key, shape_data in shapes_dict.items():
            # 检查框架溢出（文本超出形状边界）
            if shape_data.frame_overflow_bottom is not None:
                if slide_key not in overflow_map:
                    overflow_map[slide_key] = {}
                overflow_map[slide_key][shape_key] = shape_data.frame_overflow_bottom

    return overflow_map


def validate_replacements(inventory: InventoryData, replacements: dict) -> list[str]:
    """验证 replacements 中的所有形状是否存在于清单中。

    返回错误消息列表。
    """
    errors = []

    for slide_key, shapes_data in replacements.items():
        if not slide_key.startswith("slide-"):
            continue

        # 检查幻灯片是否存在
        if slide_key not in inventory:
            errors.append(f"在清单中找不到幻灯片 '{slide_key}'")
            continue

        # 检查每个形状
        for shape_key in shapes_data:
            if shape_key not in inventory[slide_key]:
                # 找出没有定义替换的形状并显示其内容
                unused_with_content = []
                for k in inventory[slide_key]:
                    if k not in shapes_data:
                        shape_data = inventory[slide_key][k]
                        # 从段落中获取文本作为预览
                        paragraphs = shape_data.paragraphs
                        if paragraphs and paragraphs[0].text:
                            first_text = paragraphs[0].text[:50]
                            if len(paragraphs[0].text) > 50:
                                first_text += "..."
                            unused_with_content.append(f"{k} ('{first_text}')")
                        else:
                            unused_with_content.append(k)

                errors.append(
                    f"在 '{slide_key}' 上找不到形状 '{shape_key}'。"
                    f"没有替换的形状：{', '.join(sorted(unused_with_content)) if unused_with_content else 'none'}"
                )

    return errors


def check_duplicate_keys(pairs):
    """加载 JSON 时检查重复键。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"在 JSON 中找到重复键：'{key}'")
        result[key] = value
    return result


def apply_replacements(pptx_file: str, json_file: str, output_file: str):
    """将 JSON 中的文本替换应用到 PowerPoint 演示文稿。"""

    # 加载演示文稿
    prs = Presentation(pptx_file)

    # 获取所有文本形状的清单（返回 ShapeData 对象）
    # 传递 prs 以使用相同的 Presentation 实例
    inventory = extract_text_inventory(Path(pptx_file), prs)

    # 检测原始演示文稿中的文本溢出
    original_overflow = detect_frame_overflow(inventory)

    # 加载替换数据，带重复键检测
    with open(json_file) as f:
        replacements = json.load(f, object_pairs_hook=check_duplicate_keys)

    # 验证替换
    errors = validate_replacements(inventory, replacements)
    if errors:
        print("错误：替换 JSON 中的形状无效：")
        for error in errors:
            print(f"  - {error}")
        print("\n请检查清单并更新您的替换 JSON。")
        print("您可以使用以下命令重新生成清单：python inventory.py <input.pptx> <output.json>")
        raise ValueError(f"发现 {len(errors)} 个验证错误")

    # 跟踪统计信息
    shapes_processed = 0
    shapes_cleared = 0
    shapes_replaced = 0

    # 从清单处理每个幻灯片
    for slide_key, shapes_dict in inventory.items():
        if not slide_key.startswith("slide-"):
            continue

        slide_index = int(slide_key.split("-")[1])

        if slide_index >= len(prs.slides):
            print(f"警告：找不到幻灯片 {slide_index}")
            continue

        # 从清单处理每个形状
        for shape_key, shape_data in shapes_dict.items():
            shapes_processed += 1

            # 直接从 ShapeData 获取形状
            shape = shape_data.shape
            if not shape:
                print(f"警告：{shape_key} 没有形状引用")
                continue

            # ShapeData 已在 __init__ 中验证 text_frame
            text_frame = shape.text_frame  # type: ignore

            text_frame.clear()  # type: ignore
            shapes_cleared += 1

            # 检查替换段落
            replacement_shape_data = replacements.get(slide_key, {}).get(shape_key, {})
            if "paragraphs" not in replacement_shape_data:
                continue

            shapes_replaced += 1

            # 添加替换段落
            for i, para_data in enumerate(replacement_shape_data["paragraphs"]):
                if i == 0:
                    p = text_frame.paragraphs[0]  # type: ignore
                else:
                    p = text_frame.add_paragraph()  # type: ignore

                apply_paragraph_properties(p, para_data)

    # 检查替换后的问题
    # 保存到临时文件并重新加载以避免在清单期间修改演示文稿
    #（extract_text_inventory 访问 font.color 会添加空的 <a:solidFill/> 元素）
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        prs.save(str(tmp_path))

    try:
        updated_inventory = extract_text_inventory(tmp_path)
        updated_overflow = detect_frame_overflow(updated_inventory)
    finally:
        tmp_path.unlink()  # 清理临时文件

    # 检查文本溢出是否恶化
    overflow_errors = []
    for slide_key, shape_overflows in updated_overflow.items():
        for shape_key, new_overflow in shape_overflows.items():
            # 获取原始溢出（如果之前没有溢出则为 0）
            original = original_overflow.get(slide_key, {}).get(shape_key, 0.0)

            # 如果溢出增加则报错
            if new_overflow > original + 0.01:  # 小的容差用于四舍五入
                increase = new_overflow - original
                overflow_errors.append(
                    f'{slide_key}/{shape_key}：溢出增加了 {increase:.2f}" '
                    f'（之前为 {original:.2f}"，现在为 {new_overflow:.2f}"）'
                )

    # 从更新的形状中收集警告
    warnings = []
    for slide_key, shapes_dict in updated_inventory.items():
        for shape_key, shape_data in shapes_dict.items():
            if shape_data.warnings:
                for warning in shape_data.warnings:
                    warnings.append(f"{slide_key}/{shape_key}：{warning}")

    # 如果有任何问题则失败
    if overflow_errors or warnings:
        print("\n错误：在替换输出中检测到问题：")
        if overflow_errors:
            print("\n文本溢出恶化：")
            for error in overflow_errors:
                print(f"  - {error}")
        if warnings:
            print("\n格式警告：")
            for warning in warnings:
                print(f"  - {warning}")
        print("\n请在保存前修复这些问题。")
        raise ValueError(f"发现 {len(overflow_errors)} 个溢出错误和 {len(warnings)} 个警告")

    # 保存演示文稿
    prs.save(output_file)

    # 报告结果
    print(f"已将更新的演示文稿保存到：{output_file}")
    print(f"处理了 {len(prs.slides)} 张幻灯片")
    print(f"  - 已处理形状：{shapes_processed}")
    print(f"  - 已清除形状：{shapes_cleared}")
    print(f"  - 已替换形状：{shapes_replaced}")


def main():
    """命令行使用的主入口点。"""
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    input_pptx = Path(sys.argv[1])
    replacements_json = Path(sys.argv[2])
    output_pptx = Path(sys.argv[3])

    if not input_pptx.exists():
        print(f"错误：找不到输入文件 '{input_pptx}'")
        sys.exit(1)

    if not replacements_json.exists():
        print(f"错误：找不到替换 JSON 文件 '{replacements_json}'")
        sys.exit(1)

    try:
        apply_replacements(str(input_pptx), str(replacements_json), str(output_pptx))
    except Exception as e:
        print(f"应用替换时出错：{e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()