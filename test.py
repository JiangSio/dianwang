"""
YOLOv8 测试/推理脚本

支持两种模式:
1. eval: 在 test 集上评估模型性能 (mAP, precision, recall)
2. infer: 对指定图片/文件夹进行检测推理，保存结果图

分片推理:
- 图片宽高均 <= TILE_SIZE 时，直接预测
- 图片任一维度 > TILE_SIZE 时，切分为子图分别预测，合并结果并还原到原图坐标
"""

import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT = "all"

# 分片配置（与训练一致）
TILE_SIZE = 1800      # 分片尺寸
OVERLAP = 200         # 重叠像素

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


def nms(boxes, scores, iou_threshold):
    """
    非极大值抑制 (NMS)
    
    Args:
        boxes: numpy array of shape (N, 4) [x1, y1, x2, y2]
        scores: numpy array of shape (N,) 置信度
        iou_threshold: IoU 阈值
    
    Returns:
        keep: 保留的索引列表
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # 按置信度降序

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


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


def draw_predictions_from_boxes(img_path, all_boxes, names, output_path):
    """
    从合并后的预测框列表绘制结果（用于分片推理）
    
    Args:
        img_path: 原图路径
        all_boxes: 列表，每个元素为 (x1, y1, x2, y2, cls_id, conf)
        names: 类别名称字典
        output_path: 输出路径
    """
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return False

    for x1, y1, x2, y2, cls_id, conf in all_boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_name = names.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)

        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # 绘制标签背景
        label = f"{cls_name}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = y1 - 10 if y1 - 10 > label_h else y1 + label_h
        cv2.rectangle(img, (x1, label_y - label_h - 5), (x1 + label_w, label_y + 5), color, -1)
        cv2.putText(img, label, (x1, label_y - 5), font, font_scale, (0, 0, 0), thickness)

    success, encoded_img = cv2.imencode('.jpg', img)
    if success:
        encoded_img.tofile(str(output_path))
    return success


def tile_predict(model, img_path, args):
    """
    分片预测：对大图切分预测，小图直接预测
    
    Returns:
        all_boxes: 列表，每个元素为 (x1, y1, x2, y2, cls_id, conf)
    """
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [错误] 无法读取图片: {img_path}")
        return []

    img_h, img_w = img.shape[:2]

    # 判断是否需要分片
    if img_w <= TILE_SIZE and img_h <= TILE_SIZE:
        # 小图，直接预测
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf_thres,
            iou=args.iou_thres,
            device=args.device,
            verbose=False,
        )
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            all_boxes = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                all_boxes.append((x1, y1, x2, y2, cls_id, conf))
            return all_boxes
        return []

    # 大图，分片预测
    x_offsets = compute_tile_offsets(img_w, TILE_SIZE, OVERLAP)
    y_offsets = compute_tile_offsets(img_h, TILE_SIZE, OVERLAP)

    all_boxes = []  # 收集所有子图的预测框（原图坐标）

    for row, y_off in enumerate(y_offsets):
        for col, x_off in enumerate(x_offsets):
            tile_x1 = x_off
            tile_y1 = y_off
            tile_x2 = min(x_off + TILE_SIZE, img_w)
            tile_y2 = min(y_off + TILE_SIZE, img_h)

            # 裁剪子图
            tile_img = img[tile_y1:tile_y2, tile_x1:tile_x2]

            # 对子图进行预测
            tile_results = model.predict(
                source=tile_img,
                imgsz=args.imgsz,
                conf=args.conf_thres,
                iou=args.iou_thres,
                device=args.device,
                verbose=False,
            )

            if tile_results and tile_results[0].boxes is not None and len(tile_results[0].boxes) > 0:
                boxes = tile_results[0].boxes
                for i in range(len(boxes)):
                    sx1, sy1, sx2, sy2 = boxes.xyxy[i].cpu().numpy()
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())

                    # 转换回原图坐标
                    orig_x1 = sx1 + tile_x1
                    orig_y1 = sy1 + tile_y1
                    orig_x2 = sx2 + tile_x1
                    orig_y2 = sy2 + tile_y1

                    all_boxes.append((orig_x1, orig_y1, orig_x2, orig_y2, cls_id, conf))

    # NMS 去重（重叠区域的同一目标可能被多个子图检测到）
    if len(all_boxes) == 0:
        return []

    # 按类别分别做 NMS
    final_boxes = []
    unique_classes = set(b[4] for b in all_boxes)
    for cls_id in unique_classes:
        cls_boxes = [b for b in all_boxes if b[4] == cls_id]
        boxes_arr = np.array([[b[0], b[1], b[2], b[3]] for b in cls_boxes])
        scores_arr = np.array([b[5] for b in cls_boxes])

        keep = nms(boxes_arr, scores_arr, args.iou_thres)
        for idx in keep:
            final_boxes.append(cls_boxes[idx])

    return final_boxes


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
    """对图片/文件夹进行推理（分片模式）"""
    print("=" * 60)
    print("YOLOv8 推理 (分片模式)")
    print("=" * 60)
    print(f"分片配置: TILE_SIZE={TILE_SIZE}, OVERLAP={OVERLAP}")

    print(f"\n模型权重: {args.weights}")
    print(f"推理源: {source}")

    # 创建输出目录
    output_dir = Path(f"runs/detect/{PROJECT}/infer")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片
    source_path = Path(source)
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = sorted(list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")))

    if not image_files:
        print("[错误] 未找到任何图片文件")
        return None

    print(f"\n找到 {len(image_files)} 张图片")

    total_boxes = 0
    for img_path in image_files:
        output_path = output_dir / f"{img_path.stem}_result.jpg"

        # 分片预测
        all_boxes = tile_predict(model, img_path, args)

        if all_boxes:
            draw_predictions_from_boxes(img_path, all_boxes, model.names, output_path)
            total_boxes += len(all_boxes)
            print(f"  {img_path.name}: {len(all_boxes)} 个目标")
            for x1, y1, x2, y2, cls_id, conf in all_boxes:
                cls_name = model.names.get(cls_id, str(cls_id))
                print(f"    - {cls_name}: {conf:.4f}")
        else:
            # 无目标也保存原图
            shutil.copy(str(img_path), str(output_path))
            print(f"  {img_path.name}: 无目标")

    print(f"\n推理完成!")
    print(f"  处理图片数: {len(image_files)}")
    print(f"  总检测目标数: {total_boxes}")
    print(f"  输出目录: {output_dir}")


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

def demo_model_init(model_path):
    weights_path = Path(model_path)

    model = YOLO(str(weights_path))
    """对图片/文件夹进行推理（分片模式）"""
    print("=" * 60)
    print("YOLOv8 推理 (分片模式)")
    print("=" * 60)
    print(f"分片配置: TILE_SIZE={TILE_SIZE}, OVERLAP={OVERLAP}")

    print(f"\n模型权重: {weights_path}")
    return model


def demo_tile_predict_img(model, img, args):
    """
    分片预测：对大图切分预测，小图直接预测
    
    Returns:
        all_boxes: 列表，每个元素为 (x1, y1, x2, y2, cls_id, conf)
    """
    if img is None:
        print(f"  [错误] 无法读取图片: {img}")
        return []

    img_h, img_w = img.shape[:2]

    # 判断是否需要分片
    if img_w <= TILE_SIZE and img_h <= TILE_SIZE:
        # 小图，直接预测
        results = model.predict(
            source=img,
            imgsz=args.imgsz,
            conf=args.conf_thres,
            iou=args.iou_thres,
            device=args.device,
            verbose=False,
        )
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            all_boxes = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                all_boxes.append((x1, y1, x2, y2, cls_id, conf))
            return all_boxes
        return []

    # 大图，分片预测
    x_offsets = compute_tile_offsets(img_w, TILE_SIZE, OVERLAP)
    y_offsets = compute_tile_offsets(img_h, TILE_SIZE, OVERLAP)

    all_boxes = []  # 收集所有子图的预测框（原图坐标）

    for row, y_off in enumerate(y_offsets):
        for col, x_off in enumerate(x_offsets):
            tile_x1 = x_off
            tile_y1 = y_off
            tile_x2 = min(x_off + TILE_SIZE, img_w)
            tile_y2 = min(y_off + TILE_SIZE, img_h)

            # 裁剪子图
            tile_img = img[tile_y1:tile_y2, tile_x1:tile_x2]

            # 对子图进行预测
            tile_results = model.predict(
                source=tile_img,
                imgsz=args.imgsz,
                conf=args.conf_thres,
                iou=args.iou_thres,
                device=args.device,
                verbose=False,
            )

            if tile_results and tile_results[0].boxes is not None and len(tile_results[0].boxes) > 0:
                boxes = tile_results[0].boxes
                for i in range(len(boxes)):
                    sx1, sy1, sx2, sy2 = boxes.xyxy[i].cpu().numpy()
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())

                    # 转换回原图坐标
                    orig_x1 = sx1 + tile_x1
                    orig_y1 = sy1 + tile_y1
                    orig_x2 = sx2 + tile_x1
                    orig_y2 = sy2 + tile_y1

                    all_boxes.append((orig_x1, orig_y1, orig_x2, orig_y2, cls_id, conf))

    # NMS 去重（重叠区域的同一目标可能被多个子图检测到）
    if len(all_boxes) == 0:
        return []

    # 按类别分别做 NMS
    final_boxes = []
    unique_classes = set(b[4] for b in all_boxes)
    for cls_id in unique_classes:
        cls_boxes = [b for b in all_boxes if b[4] == cls_id]
        boxes_arr = np.array([[b[0], b[1], b[2], b[3]] for b in cls_boxes])
        scores_arr = np.array([b[5] for b in cls_boxes])

        keep = nms(boxes_arr, scores_arr, args.iou_thres)
        for idx in keep:
            final_boxes.append(cls_boxes[idx])

    return final_boxes

def demo_infer(model,test_img):
    
    args = parse_args()
    all_boxes = demo_tile_predict_img(model, test_img, args)
    print(all_boxes)
    return all_boxes

        

if __name__ == "__main__":
    main()
