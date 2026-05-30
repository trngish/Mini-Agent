#!/usr/bin/env python3
"""
技能打包脚本 - 将技能文件夹打包成可分发的 zip 文件

用法：
    python utils/package_skill.py <path/to/skill-folder> [output-directory]

示例：
    python utils/package_skill.py skills/public/my-skill
    python utils/package_skill.py skills/public/my-skill ./dist
"""

import sys
import zipfile
from pathlib import Path

from quick_validate import validate_skill


def package_skill(skill_path, output_dir=None):
    """
    将技能文件夹打包成 zip 文件。

    参数：
        skill_path: 技能文件夹路径
        output_dir: 可选的 zip 文件输出目录（默认为当前目录）

    返回：
        创建的 zip 文件路径，出错则返回 None
    """
    skill_path = Path(skill_path).resolve()

    # 验证技能文件夹是否存在
    if not skill_path.exists():
        print(f"❌ 错误：未找到技能文件夹：{skill_path}")
        return None

    if not skill_path.is_dir():
        print(f"❌ 错误：路径不是目录：{skill_path}")
        return None

    # 验证 SKILL.md 是否存在
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        print(f"❌ 错误：未在 {skill_path} 中找到 SKILL.md")
        return None

    # 打包前运行验证
    print("🔍 正在验证技能...")
    valid, message = validate_skill(skill_path)
    if not valid:
        print(f"❌ Validation failed: {message}")
        print("   Please fix the validation errors before packaging.")
        return None
    print(f"✅ {message}\n")

    # 确定输出位置
    skill_name = skill_path.name
    if output_dir:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd()

    zip_filename = output_path / f"{skill_name}.zip"

    # 创建 zip 文件
    try:
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            # 遍历技能目录
            for file_path in skill_path.rglob("*"):
                if file_path.is_file():
                    # 计算 zip 内的相对路径
                    arcname = file_path.relative_to(skill_path.parent)
                    zipf.write(file_path, arcname)
                    print(f"  已添加：{arcname}")

        print(f"\n✅ Successfully packaged skill to: {zip_filename}")
        return zip_filename

    except Exception as e:
        print(f"❌ Error creating zip file: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("用法：python utils/package_skill.py <path/to/skill-folder> [output-directory]")
        print("\n示例：")
        print("  python utils/package_skill.py skills/public/my-skill")
        print("  python utils/package_skill.py skills/public/my-skill ./dist")
        sys.exit(1)

    skill_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"📦 正在打包技能：{skill_path}")
    if output_dir:
        print(f"   Output directory: {output_dir}")
    print()

    result = package_skill(skill_path, output_dir)

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
