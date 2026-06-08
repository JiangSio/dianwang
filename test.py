"""
YOLOv8 测试/推理脚本

支持两种模式:
1. eval: 在 test 集上评估模型性能 (mAP, precision, recall)
2. infer: 对指定图片/文件夹进行检测推理，保存结果图
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def evaluate(model, data, args):
    """在 test 集上评估模型"""
    print("=" * 60)
    print("YOLOv8 模型评估")
    print("=" * 60)

    print(f"\n模型权重: {args.weights}")
    print(f"数据集: {data}")

    # 执行验证
    results = model.val(
        data=data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=True,
        save_json=True,
        save_hybrid=False,
    )

    # 打印关键指标
    print(f"\n评估结果:")
    print(f"  mAP50:    {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall:    {results.box.mr:.4f}")

    # 各类别指标
    print(f"\n各类别指标:")
    names = model.names
    for i, name in names.items():
        print(f"  {name} (class {i}):")
        if hasattr(results.box, 'ap50') and len(results.box.ap50) > i:
            print(f"    AP50: {results.box.ap50[i]:.4f}")
        if hasattr(results.box, 'ap') and len(results.box.ap) > i:
            print(f"    AP:   {results.box.ap[i]:.4f}")

    return results


def infer(model, source, args):
    """对图片/文件夹进行推理"""
    print("=" * 60)
    print("YOLOv8 推理")
    print("=" * 60)

    print(f"\n模型权重: {args.weights}")
    print(f"推理源: {source}")

    # 执行推理
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf_thres,
        iou=args.iou_thres,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    # 打印结果摘要
    print(f"\n推理完成!")
    print(f"  处理图片数: {len(results)}")
    print(f"  结果保存至: {Path(args.project) / args.name}")

    # 打印每张图片的检测结果
    for r in results:
        path = Path(r.path).name
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"  {path}: {len(boxes)} 个目标")
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                cls_name = model.names.get(cls_id, str(cls_id))
                print(f"    - {cls_name}: {conf:.4f}")
        else:
            print(f"  {path}: 无目标")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 测试/推理脚本")

    # 通用参数
    parser.add_argument("--mode", type=str, choices=["eval", "infer"],
                        default="eval", help="运行模式: eval(评估) 或 infer(推理)")
    parser.add_argument("--weights", type=str, default="weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--data", type=str, default="dataset.yaml",
                        help="数据集配置文件 (仅 eval 模式)")
    parser.add_argument("--source", type=str, default="data/images/test",
                        help="推理源: 图片/文件夹路径 (仅 infer 模式)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size")
    parser.add_argument("--device", type=str, default="",
                        help="设备 (0/cpu/0,1,2,3)")

    # 推理专用参数
    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="置信度阈值 (仅 infer 模式)")
    parser.add_argument("--iou_thres", type=float, default=0.45,
                        help="NMS IoU 阈值 (仅 infer 模式)")
    parser.add_argument("--save_txt", action="store_true",
                        help="保存检测结果为 TXT (仅 infer 模式)")
    parser.add_argument("--project", type=str, default="runs/predict",
                        help="推理结果保存目录")
    parser.add_argument("--name", type=str, default="exp",
                        help="推理实验名称")

    return parser.parse_args()


def main():
    args = parse_args()

    # 验证权重文件存在
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")

    # 加载模型
    print(f"加载模型: {args.weights}")
    model = YOLO(str(weights_path))

    if args.mode == "eval":
        evaluate(model, args.data, args)
    elif args.mode == "infer":
        infer(model, args.source, args)


if __name__ == "__main__":
    main()
