"""
数据集划分脚本

将 data/images/ 和 data/labels/ 下的文件划分为 train / val / test 三个子集

划分比例: train 70% / val 15% / test 15%
"""

import os
import random
import shutil
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent
PROJECT = "luoshuan"
DATA_DIR = BASE_DIR / "data" / PROJECT
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

# 划分比例
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# 随机种子 (保证可复现)
RANDOM_SEED = 42


def split_dataset():
    """执行数据集划分"""
    print("=" * 60)
    print("数据集划分 (train/val/test)")
    print("=" * 60)

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
    # test 使用剩余部分，确保总和一致
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

    # 清理空的根目录文件
    _clean_empty_dirs()

    print(f"\n划分完成!")
    print(f"  输出目录: {DATA_DIR}")


def _clean_empty_dirs():
    """清理可能为空的根级 images/labels 目录中的残留文件"""
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
        img_count = len(list((IMAGES_DIR / split).glob("*.jpg")))
        label_count = len(list((LABELS_DIR / split).glob("*.txt")))
        print(f"  {split}: {img_count} images, {label_count} labels")


if __name__ == "__main__":
    main()
