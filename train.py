"""
YOLOv8 训练脚本

使用预训练模型在自定义数据集上进行目标检测训练
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

PROJECT = "luoshuan"

# 默认训练参数
DEFAULT_CONFIG = {
    "model": "yolov8n.pt",
    "data": f"dataset_{PROJECT}.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "",
    "workers": 4,
    "lr0": 0.01,
    "patience": 50,
    "project": f"{PROJECT}",
    "name": "train",
    "optimizer": "adamw",
    "close_mosaic": 10,
    "amp": True,
    "cache": False,
}


def train(args):
    """执行训练"""
    print("=" * 60)
    print("YOLOv8 训练")
    print("=" * 60)

    # 验证数据集配置文件存在
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_path}")

    # 加载预训练模型
    print(f"\n加载模型: {args.model}")
    model = YOLO(args.model)

    # 训练参数
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "lr0": args.lr0,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "optimizer": args.optimizer,
        "close_mosaic": args.close_mosaic,
        "amp": args.amp,
        "cache": args.cache,
        "verbose": True,
        "save": True,
        "save_period": 10,
        "exist_ok": True,
    }

    print(f"\n训练参数:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")

    # 开始训练
    print(f"\n开始训练...")
    results = model.train(**train_args)

    print(f"\n训练完成!")
    print(f"  结果目录: {Path(args.project) / args.name}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model"],
                        help="预训练模型路径 (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--data", type=str, default=DEFAULT_CONFIG["data"],
                        help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=DEFAULT_CONFIG["batch"],
                        help="batch size")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_CONFIG["imgsz"],
                        help="输入图像尺寸")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"],
                        help="设备 (0/cpu/0,1,2,3)")
    parser.add_argument("--workers", type=int, default=DEFAULT_CONFIG["workers"],
                        help="数据加载线程数")
    parser.add_argument("--lr0", type=float, default=DEFAULT_CONFIG["lr0"],
                        help="初始学习率")
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"],
                        help="早停轮数")
    parser.add_argument("--optimizer", type=str, default=DEFAULT_CONFIG["optimizer"],
                        help="优化器 (auto/adamw)")
    parser.add_argument("--project", type=str, default=DEFAULT_CONFIG["project"],
                        help="结果保存目录")
    parser.add_argument("--name", type=str, default=DEFAULT_CONFIG["name"],
                        help="实验名称")
    parser.add_argument("--close_mosaic", type=int, default=DEFAULT_CONFIG["close_mosaic"],
                        help="最后 N 轮关闭 mosaic 增强")
    parser.add_argument("--amp", type=bool, default=DEFAULT_CONFIG["amp"],
                        help="启用混合精度训练 (True/False)")
    parser.add_argument("--cache", type=bool, default=DEFAULT_CONFIG["cache"],
                        help="缓存图像到内存加速训练 (True/False)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
