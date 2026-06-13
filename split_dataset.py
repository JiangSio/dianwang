"""
数据集划分 + 分片脚本

1. 将 data/images/ 和 data/labels/ 下的文件划分为 train / val / test 三个子集
2. 对每个子集执行分片，生成 _cut 子图数据集用于训练
3. 保存偏移信息到 tile_offsets/ 目录

划分比例: train 90% / val 5% / test 5%
分片配置: TILE_SIZE=1000, OVERLAP=200
"""

import os
import random
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent
PROJECT = "luoshuan"
DATA_DIR = BASE_DIR / "data" / PROJECT
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
TILE_OFFSETS_DIR = DATA_DIR / "tile_offsets"

# 划分比例
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# 分片配置
TILE_SIZE = 1000      # 分片尺寸
OVERLAP = 200         # 重叠像素

# 随机种子
RANDOM_SEED = 42


def compute_tile_offsets(img_size, tile_size, overlap):
    """计算分片的起始偏移量列表（含重叠）"""
    offsets = []
    step = tile_size - overlap
    pos = 0
    while pos < img_size:
        offsets.append(pos)
        pos += step
        if pos + tile_size > img_size and pos < img_size:
            offsets.append(max(0, img_size - tile_size))
            break
    return offsets


def clip_bbox(xmin, ymin, xmax, ymax, w, h):
    """将标注框裁剪到图片范围内"""
    xmin = max(0, min(xmin, w))
    ymin = max(0, min(ymin, h))
    xmax = max(0, min(xmax, w))
    ymax = max(0, min(ymax, h))
    return xmin, ymin, xmax, ymax


def split_image_and_labels(img_path, label_path, cut_images_dir, cut_labels_dir, tile_mapping):
    """
    对单张图片执行分片
    
    Returns: (子图数量, 标注框总数)
    """
    # 读取图片
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [错误] 无法读取图片: {img_path}")
        return 0, 0

    img_h, img_w = img.shape[:2]
    stem = img_path.stem

    # 读取YOLO标注
    yolo_boxes = []  # [(cls_id, xc, yc, w, h), ...]
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    yolo_boxes.append((cls_id, xc, yc, w, h))

    # 判断是否需要分片
    if img_w <= TILE_SIZE and img_h <= TILE_SIZE:
        # 小图，直接复制
        shutil.copy2(img_path, cut_images_dir / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, cut_labels_dir / label_path.name)
        else:
            # 创建空标注文件
            (cut_labels_dir / label_path.name).touch()

        tile_mapping[img_path.name] = {
            "original": img_path.name,
            "offset_x": 0,
            "offset_y": 0,
            "tile_w": img_w,
            "tile_h": img_h,
        }
        return 1, len(yolo_boxes)

    # 大图，需要分片
    x_offsets = compute_tile_offsets(img_w, TILE_SIZE, OVERLAP)
    y_offsets = compute_tile_offsets(img_h, TILE_SIZE, OVERLAP)

    tile_count = 0
    total_objects = 0

    for row, y_off in enumerate(y_offsets):
        for col, x_off in enumerate(x_offsets):
            tile_x1 = x_off
            tile_y1 = y_off
            tile_x2 = min(x_off + TILE_SIZE, img_w)
            tile_y2 = min(y_off + TILE_SIZE, img_h)
            tile_w = tile_x2 - tile_x1
            tile_h = tile_y2 - tile_y1

            # 裁剪子图
            tile_img = img[tile_y1:tile_y2, tile_x1:tile_x2]

            # 筛选落在该子图内的标注框（中心点在子图内）
            tile_boxes = []
            for cls_id, xc, yc, w, h in yolo_boxes:
                # 反归一化得到原图绝对坐标
                abs_xc = xc * img_w
                abs_yc = yc * img_h
                abs_w = w * img_w
                abs_h = h * img_h

                abs_xmin = abs_xc - abs_w / 2
                abs_ymin = abs_yc - abs_h / 2
                abs_xmax = abs_xc + abs_w / 2
                abs_ymax = abs_yc + abs_h / 2

                # 判断中心点是否在子图内
                if not (tile_x1 <= abs_xc <= tile_x2 and tile_y1 <= abs_yc <= tile_y2):
                    continue

                # 转换到子图相对坐标
                sub_xmin = abs_xmin - tile_x1
                sub_ymin = abs_ymin - tile_y1
                sub_xmax = abs_xmax - tile_x1
                sub_ymax = abs_ymax - tile_y1

                # 裁剪到子图范围内
                sub_xmin, sub_ymin, sub_xmax, sub_ymax = clip_bbox(
                    sub_xmin, sub_ymin, sub_xmax, sub_ymax, tile_w, tile_h
                )

                # 跳过裁剪后面积为0的框
                if sub_xmax - sub_xmin <= 0 or sub_ymax - sub_ymin <= 0:
                    continue

                # 重新归一化（相对于子图尺寸）
                sub_xc = (sub_xmin + sub_xmax) / 2 / tile_w
                sub_yc = (sub_ymin + sub_ymax) / 2 / tile_h
                sub_w = (sub_xmax - sub_xmin) / tile_w
                sub_h = (sub_ymax - sub_ymin) / tile_h

                tile_boxes.append((cls_id, sub_xc, sub_yc, sub_w, sub_h))

            if not tile_boxes:
                continue  # 无标注的子图跳过

            # 保存子图
            tile_name = f"{stem}_tile_{row}_{col}.jpg"
            out_img = cut_images_dir / tile_name
            out_label = cut_labels_dir / f"{stem}_tile_{row}_{col}.txt"

            success, encoded = cv2.imencode('.jpg', tile_img)
            if success:
                encoded.tofile(str(out_img))
            else:
                print(f"  [错误] 保存子图失败: {tile_name}")
                continue

            # 保存YOLO标注
            lines = []
            for cls_id, xc, yc, w, h in tile_boxes:
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            with open(out_label, "w") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")

            # 记录映射
            tile_mapping[tile_name] = {
                "original": img_path.name,
                "offset_x": tile_x1,
                "offset_y": tile_y1,
                "tile_w": tile_w,
                "tile_h": tile_h,
            }

            tile_count += 1
            total_objects += len(tile_boxes)

    return tile_count, total_objects


def split_dataset():
    """执行数据集划分 + 分片"""
    print("=" * 60)
    print("数据集划分 + 分片 (train/val/test)")
    print("=" * 60)
    print(f"分片配置: TILE_SIZE={TILE_SIZE}, OVERLAP={OVERLAP}")

    # 获取所有图片文件
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    total = len(image_files)

    if total == 0:
        print("[错误] 未找到任何图片文件，请先运行 convert_voc2yolo.py")
        return

    print(f"\n总样本数: {total}")
    print(f"划分比例: train={TRAIN_RATIO:.0%}, val={VAL_RATIO:.0%}, test={TEST_RATIO:.0%}")

    # 打乱并划分
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    print(f"实际划分: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # 创建目标目录
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, files in splits.items():
        split_images_dir = IMAGES_DIR / split_name
        split_labels_dir = LABELS_DIR / split_name
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            stem = img_path.stem
            label_path = LABELS_DIR / f"{stem}.txt"

            # 移动图片
            shutil.move(str(img_path), str(split_images_dir / img_path.name))

            # 移动标注 (如果存在)
            if label_path.exists():
                shutil.move(str(label_path), str(split_labels_dir / label_path.name))

    # 清理根目录残留文件
    _clean_root_files()

    print(f"\n原始数据集划分完成!")

    # 对每个子集执行分片
    print(f"\n开始分片...")
    TILE_OFFSETS_DIR.mkdir(parents=True, exist_ok=True)

    total_orig_images = 0
    total_cut_images = 0
    total_cut_objects = 0

    for split_name in ["train", "val", "test"]:
        split_images_dir = IMAGES_DIR / split_name
        split_labels_dir = LABELS_DIR / split_name
        cut_images_dir = IMAGES_DIR / f"{split_name}_cut"
        cut_labels_dir = LABELS_DIR / f"{split_name}_cut"

        cut_images_dir.mkdir(parents=True, exist_ok=True)
        cut_labels_dir.mkdir(parents=True, exist_ok=True)

        tile_mapping = {}
        orig_count = 0
        cut_count = 0
        cut_objects = 0

        orig_images = sorted(split_images_dir.glob("*.jpg"))
        orig_count = len(orig_images)

        for img_path in orig_images:
            stem = img_path.stem
            label_path = split_labels_dir / f"{stem}.txt"

            tc, to = split_image_and_labels(
                img_path, label_path, cut_images_dir, cut_labels_dir, tile_mapping
            )
            cut_count += tc
            cut_objects += to

        # 保存偏移信息
        offset_file = TILE_OFFSETS_DIR / f"{split_name}_cut.json"
        with open(offset_file, "w") as f:
            json.dump(tile_mapping, f, indent=2, ensure_ascii=False)

        print(f"  {split_name}: {orig_count} 原图 -> {cut_count} 子图, {cut_objects} 标注框")

        total_orig_images += orig_count
        total_cut_images += cut_count
        total_cut_objects += cut_objects

    print(f"\n分片完成!")
    print(f"  总原图数: {total_orig_images}")
    print(f"  总子图数: {total_cut_images}")
    print(f"  总标注框数: {total_cut_objects}")


def _clean_root_files():
    """清理根级 images/labels 目录中的残留文件"""
    for d in [IMAGES_DIR, LABELS_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()


def main():
    split_dataset()

    # 验证结果
    print("\n验证划分结果:")
    for split in ["train", "val", "test"]:
        orig_img = len(list((IMAGES_DIR / split).glob("*.jpg")))
        orig_label = len(list((LABELS_DIR / split).glob("*.txt")))
        cut_img = len(list((IMAGES_DIR / f"{split}_cut").glob("*.jpg")))
        cut_label = len(list((LABELS_DIR / f"{split}_cut").glob("*.txt")))
        print(f"  {split}: 原图 {orig_img} images, {orig_label} labels | 子图 {cut_img} images, {cut_label} labels")


if __name__ == "__main__":
    main()
