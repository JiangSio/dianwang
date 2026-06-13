"""
可视化 VOC2007 XML 标注文件

输入: 法兰盘连接螺栓缺失/ 目录下的 .xml 和 .jpg 文件
输出: visualization/ 目录下带有标注框的图片
"""

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import shutil
from pathlib import Path

# 类别颜色映射 (每个类别不同颜色)
CLASS_MAP_1 = {
    "07010001": (255, 0, 0),
    "07010002": (0, 255, 0),
}
CLASS_MAP_2 = {
    "07011027": (0, 255, 0),  # 绿色
}
PROJECT_NAME = "法兰盘连接螺栓缺失"
PROJECT_NAME = "挂点金具开口销缺失"
CLASS_COLORS = CLASS_MAP_1
# 路径配置
BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / PROJECT_NAME
OUTPUT_DIR = BASE_DIR / f"{PROJECT_NAME}_标注"


def parse_voc_xml(xml_path):
    """解析 VOC XML 文件，返回 (width, height, [(class_name, xmin, ymin, xmax, ymax)])"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        objects.append((class_name, xmin, ymin, xmax, ymax))

    return width, height, objects


def draw_annotations(img_path, objects, output_path):
    """在图片上绘制标注框"""
    # 使用 cv2.imdecode 支持中文路径
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [错误] 无法读取图片: {img_path}")
        return False

    # 绘制每个标注框
    for class_name, xmin, ymin, xmax, ymax in objects:
        color = CLASS_COLORS.get(class_name, (0, 0, 255))  # 默认红色
        
        # 绘制矩形框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
        
        # 绘制标签背景
        label = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 标签位置（在框上方）
        label_y = ymin - 10 if ymin - 10 > label_h else ymin + label_h
        cv2.rectangle(img, (xmin, label_y - label_h - 5), (xmin + label_w, label_y + 5), color, -1)
        
        # 绘制标签文字
        cv2.putText(img, label, (xmin, label_y - 5), font, font_scale, (0, 0, 0), thickness)

    # 使用 cv2.imencode 支持中文路径保存
    success, encoded_img = cv2.imencode('.jpg', img)
    if success:
        encoded_img.tofile(str(output_path))
    return success


def main():
    print("=" * 60)
    print("VOC2007 标注可视化")
    print("=" * 60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 获取所有 XML 文件
    xml_files = sorted(SOURCE_DIR.glob("*.xml"))
    print(f"\n找到 {len(xml_files)} 个 XML 文件")

    success_count = 0
    skipped_count = 0

    for xml_path in xml_files:
        stem = xml_path.stem
        jpg_path = SOURCE_DIR / f"{stem}.jpg"

        if not jpg_path.exists():
            print(f"  [跳过] 找不到对应图片: {stem}")
            skipped_count += 1
            continue

        # 解析标注
        width, height, objects = parse_voc_xml(xml_path)
        
        if not objects:
            print(f"  [跳过] 无标注对象: {stem}")
            skipped_count += 1
            continue

        # 输出路径
        output_path = OUTPUT_DIR / f"{stem}_visualized.jpg"

        # 绘制标注
        if draw_annotations(jpg_path, objects, output_path):
            success_count += 1
            print(f"  [成功] {stem} - {len(objects)} 个标注框")

    print(f"\n可视化完成!")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
