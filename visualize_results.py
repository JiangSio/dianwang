"""
训练结果可视化脚本

功能:
1. 绘制训练 loss/mAP 曲线
2. 展示测试集检测样例
3. 显示混淆矩阵 (如可用)
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 默认路径配置
DEFAULT_RESULTS_DIR = "runs/detect/train"
DEFAULT_PREDICT_DIR = "runs/predict/exp"


def plot_metrics(results_dir):
    """绘制训练指标曲线"""
    print("=" * 60)
    print("训练指标可视化")
    print("=" * 60)

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[错误] 结果目录不存在: {results_path}")
        print("请先运行训练脚本")
        return

    # 查找结果图片
    result_images = {
        "results.png": "训练指标 (mAP & Loss)",
        "PR_curve.png": "Precision-Recall 曲线",
        "F1_curve.png": "F1-Confidence 曲线",
        "confusion_matrix.png": "混淆矩阵",
        "confusion_matrix_normalized.png": "归一化混淆矩阵",
    }

    found = False
    for filename, title in result_images.items():
        img_path = results_path / filename
        if img_path.exists():
            found = True
            print(f"\n显示: {title}")
            fig, ax = plt.subplots(figsize=(10, 8))
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()
        else:
            print(f"  [跳过] 未找到: {filename}")

    if not found:
        print("\n未找到任何训练结果图片")
        print(f"请检查目录: {results_path}")


def plot_detection_results(predict_dir):
    """展示推理结果图片"""
    print("\n" + "=" * 60)
    print("检测结果可视化")
    print("=" * 60)

    predict_path = Path(predict_dir)
    if not predict_path.exists():
        print(f"[错误] 推理结果目录不存在: {predict_path}")
        print("请先运行推理: python test.py --mode infer")
        return

    # 获取所有结果图片
    result_images = sorted(predict_path.glob("*.jpg")) + sorted(predict_path.glob("*.png"))

    if not result_images:
        print(f"\n未找到推理结果图片")
        print(f"请检查目录: {predict_path}")
        return

    print(f"\n找到 {len(result_images)} 张结果图片")

    # 显示前 9 张
    max_show = 9
    images_to_show = result_images[:max_show]

    cols = min(3, len(images_to_show))
    rows = (len(images_to_show) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, img_path in enumerate(images_to_show):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]

        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_title(img_path.name, fontsize=10)
        ax.axis("off")

    # 隐藏多余子图
    for idx in range(len(images_to_show), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="训练结果可视化脚本")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="训练结果目录")
    parser.add_argument("--predict_dir", type=str, default=DEFAULT_PREDICT_DIR,
                        help="推理结果目录")
    parser.add_argument("--mode", type=str, choices=["metrics", "detection", "all"],
                        default="all", help="可视化模式")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in ["metrics", "all"]:
        plot_metrics(args.results_dir)

    if args.mode in ["detection", "all"]:
        plot_detection_results(args.predict_dir)


if __name__ == "__main__":
    main()
