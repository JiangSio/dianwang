"""
YOLOv8 测试/推理脚本

支持两种模式:
1. eval: 在 test 集上评估模型性能 (mAP, precision, recall)
2. infer: 对指定图片/文件夹进行检测推理，保存结果图
"""

import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT = "luoshuan"

# 类别颜色映射 (BGR 格式)
CLASS_COLORS = {
    0: (0, 255, 0),    # 绿色
    1: (255, 0, 0),    # 蓝色
    2: (0, 0, 255),    # 红色
    3: (255, 255, 0),  # 青色
    4: (255, 0, 255),  # 品红
    5: (0, 255, 255),  # 黄色
    6: (128, 0, 128),  # 紫色
    7: (255, 165, 0),  # 橙色
    8: (128, 128, 0),  # 橄榄色
    9: (0, 128, 128),  # 蓝绿色
}
DEFAULT_COLOR = (200, 200, 200)


def draw_predictions(img_path, boxes, names, output_path):
    """在图片上绘制预测框 (与 visualize_ground.py 统一风格)"""
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return False

    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        cls_name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)

        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # 绘制标签背景
        label = f"{cls_name}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # 标签位置（在框上方）
        label_y = y1 - 10 if y1 - 10 > label_h else y1 + label_h
        cv2.rectangle(img, (x1, label_y - label_h - 5), (x1 + label_w, label_y + 5), color, -1)

        # 绘制标签文字 (黑色)
        cv2.putText(img, label, (x1, label_y - 5), font, font_scale, (0, 0, 0), thickness)

    # 保存
    success, encoded_img = cv2.imencode('.jpg', img)
    if success:
        encoded_img.tofile(str(output_path))
    return success

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
        project=f"{PROJECT}",
        name="test",
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

    # 创建输出目录
    output_dir = Path(f"runs/detect/{PROJECT}/infer")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 执行推理 (不自动保存，手动绘制)
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf_thres,
        iou=args.iou_thres,
        device=args.device,
        save=False,  # 不自动保存
        save_txt=args.save_txt,
        project=f"{PROJECT}",
        name="infer",
        exist_ok=True,
        verbose=True,
    )

    # 手动绘制并保存
    print(f"\n保存检测结果...")
    for r in results:
        path = Path(r.path)
        output_path = output_dir / f"{path.stem}_result.jpg"
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            draw_predictions(path, boxes, model.names, output_path)
            print(f"  {path.name}: {len(boxes)} 个目标")
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                cls_name = model.names.get(cls_id, str(cls_id))
                print(f"    - {cls_name}: {conf:.4f}")
        else:
            # 无目标也保存原图
            shutil.copy(str(path), str(output_path))
            print(f"  {path.name}: 无目标")

    print(f"\n推理完成!")
    print(f"  处理图片数: {len(results)}")
    print(f"  输出目录: {output_dir}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 测试/推理脚本")

    # 通用参数
    parser.add_argument("--mode", type=str, choices=["eval", "infer"],
                        default="eval", help="运行模式: eval(评估) 或 infer(推理)")
    parser.add_argument("--weights", type=str, default=f"runs/detect/{PROJECT}/train/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--data", type=str, default=f"dataset_{PROJECT}.yaml",
                        help="数据集配置文件 (仅 eval 模式)")
    parser.add_argument("--source", type=str, default=f"data/{PROJECT}/images/test",
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
