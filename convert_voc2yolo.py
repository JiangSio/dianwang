"""
VOC2007 XML 标注转换为 YOLO 格式 TXT 标注

输入: 挂点金具开口销缺失/ 目录下的 .xml 和 .jpg 文件
输出: data/images/ 和 data/labels/ 目录下的对应文件
"""

import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

# 类别映射 (VOC class name -> YOLO class id)
CLASS_MAP_1 = {
    "07010001": 0,
    "07010002": 1,
}
CLASS_MAP_2 = {
    "07011027": 0,
}
CLASS_MAP_ALL = {
    "07010001": 0,
    "07010002": 1,
    "07011027": 2,
}

CLASS_MAP = CLASS_MAP_ALL

# 路径配置
BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR / ".." /"all"
OUTPUT_DIR = BASE_DIR / "data" / "all"
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"


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
        if class_name not in CLASS_MAP:
            print(f"  [警告] 未知类别: {class_name}, 跳过")
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        objects.append((class_name, xmin, ymin, xmax, ymax))

    return width, height, objects


def voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_width, img_height):
    """将 VOC 边界框转换为 YOLO 格式 (归一化的 x_center, y_center, width, height)"""
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height

    # 裁剪到 [0, 1] 范围
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return x_center, y_center, w, h


def convert_single_xml(xml_path, output_label_path):
    """转换单个 XML 文件为 YOLO TXT 格式"""
    img_width, img_height, objects = parse_voc_xml(xml_path)

    lines = []
    for class_name, xmin, ymin, xmax, ymax in objects:
        class_id = CLASS_MAP[class_name]
        x_center, y_center, w, h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_width, img_height)
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    with open(output_label_path, "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "")

    return len(objects)


def main():
    print("=" * 60)
    print("VOC2007 XML -> YOLO TXT 格式转换")
    print("=" * 60)

    # 创建输出目录
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 获取所有 XML 文件
    xml_files = sorted(SOURCE_DIR.glob("*.xml"))
    print(f"\n找到 {len(xml_files)} 个 XML 文件")

    total_objects = 0
    skipped = 0

    for xml_path in xml_files:
        # 对应的图片文件名
        stem = xml_path.stem
        jpg_path = SOURCE_DIR / f"{stem}.jpg"

        if not jpg_path.exists():
            print(f"  [跳过] 找不到对应图片: {stem}")
            skipped += 1
            continue

        # 输出路径
        output_label = LABELS_DIR / f"{stem}.txt"
        output_image = IMAGES_DIR / f"{stem}.jpg"

        # 转换标注
        num_objects = convert_single_xml(xml_path, output_label)
        total_objects += num_objects

        # 复制图片
        shutil.copy2(jpg_path, output_image)

    print(f"\n转换完成!")
    print(f"  成功: {len(xml_files) - skipped}")
    print(f"  跳过: {skipped}")
    print(f"  总标注框数: {total_objects}")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
