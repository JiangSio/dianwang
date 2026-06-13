"""
基于 Gradio 的训练结果可视化 Web 应用

功能:
1. 交互式浏览训练指标曲线
2. 交互式浏览检测结果图片
3. 预测结果与 Ground Truth 同步对比查看
4. 支持服务器远程访问 (通过浏览器)

使用方法:
    python visualize_gradio.py                    # 默认端口 7860
    python visualize_gradio.py --port 8080        # 自定义端口
    python visualize_gradio.py --share            # 生成公网链接
"""

import os

# 禁用代理，避免 Gradio 启动时通过代理访问 localhost
for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(var, None)

import gradio as gr
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
import cv2
import numpy as np


# 默认路径配置
PROJECT = "luoshuan"
DEFAULT_RESULTS_DIR = f"runs/detect/{PROJECT}/train"
DEFAULT_PREDICT_DIR = f"runs/detect/{PROJECT}/infer"
DEFAULT_GT_DIR = f"data/{PROJECT}/images/test"
DEFAULT_LABELS_DIR = f"data/{PROJECT}/labels/test"


# ==================== 训练指标相关 ====================

def get_metric_images(results_dir):
    """获取训练指标图片列表"""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    metric_files = [
        "results.png",
        "PR_curve.png",
        "F1_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    ]

    available = []
    for f in metric_files:
        path = results_path / f
        if path.exists():
            available.append(str(path))

    return available


def display_metric(results_dir, metric_name):
    """显示选定的训练指标图片"""
    if not results_dir or not Path(results_dir).exists():
        return None, "训练结果目录不存在"

    img_path = Path(results_dir) / metric_name
    if img_path.exists():
        return str(img_path), f"显示: {metric_name}"
    return None, f"未找到: {metric_name}"


def update_metric_dropdown(results_dir):
    """更新指标下拉选项"""
    if not results_dir or not Path(results_dir).exists():
        return gr.Dropdown(choices=[], value=None)

    metric_files = [
        "results.png",
        "PR_curve.png",
        "F1_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    ]

    available = [f for f in metric_files if (Path(results_dir) / f).exists()]
    if available:
        return gr.Dropdown(choices=available, value=available[0])
    return gr.Dropdown(choices=[], value=None)


# ==================== 检测结果浏览 ====================

def get_detection_images(predict_dir):
    """获取检测结果图片列表"""
    predict_path = Path(predict_dir)
    if not predict_path.exists():
        return []

    images = sorted(predict_path.glob("*.jpg")) + sorted(predict_path.glob("*.png"))
    return [str(p) for p in images]


def get_detection_max(predict_dir):
    """获取检测结果图片最大索引"""
    images = get_detection_images(predict_dir)
    return len(images) - 1 if images else 0


def display_detection(predict_dir, image_index):
    """显示选定的检测结果图片"""
    if not predict_dir or not Path(predict_dir).exists():
        return None, "推理结果目录不存在", gr.Slider(minimum=0, maximum=1, value=0, step=1)

    images = get_detection_images(predict_dir)
    if not images:
        return None, "未找到推理结果图片", gr.Slider(minimum=0, maximum=1, value=0, step=1)

    max_idx = len(images) - 1
    idx = int(image_index) % len(images)
    return images[idx], f"图片 {idx + 1}/{len(images)}", gr.Slider(minimum=0, maximum=max_idx, value=idx, step=1)


def update_detection_slider(predict_dir):
    """更新检测结果滑块"""
    images = get_detection_images(predict_dir)
    if images:
        return gr.Slider(minimum=0, maximum=len(images) - 1, value=0, step=1)
    return gr.Slider(minimum=0, maximum=1, value=0, step=1)


# ==================== 预测 vs Ground Truth 对比 ====================

def get_image_pairs(predict_dir, gt_dir):
    """获取预测结果和 GT 的图片对"""
    predict_path = Path(predict_dir)
    gt_path = Path(gt_dir)

    if not predict_path.exists():
        return [], "预测结果目录不存在"
    if not gt_path.exists():
        return [], "Ground Truth 目录不存在"

    # 获取预测图片文件名
    predict_images = sorted(predict_path.glob("*.jpg")) + sorted(predict_path.glob("*.png"))
    predict_names = {p.stem: str(p) for p in predict_images}

    # 获取 GT 图片文件名 (支持 jpg/png)
    gt_images = sorted(gt_path.glob("*.jpg")) + sorted(gt_path.glob("*.png"))
    gt_names = {p.stem: str(p) for p in gt_images}

    # 匹配共同的文件名
    common_names = sorted(set(predict_names.keys()) & set(gt_names.keys()))

    pairs = []
    for name in common_names:
        pairs.append({
            "name": name,
            "predict": predict_names[name],
            "gt": gt_names[name],
        })

    return pairs, f"找到 {len(pairs)} 对匹配图片"


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

# 默认颜色 (超出预定义类别时使用)
DEFAULT_COLOR = (200, 200, 200)  # 灰色


def draw_bboxes(img, bboxes, label_prefix=""):
    """在图片上绘制标注框 (统一风格: 框 + 标签背景 + 黑色文字)"""
    for cls_id, xmin, ymin, xmax, ymax in bboxes:
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))

        # 绘制矩形框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)

        # 绘制标签背景
        label = f"{label_prefix}{cls_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # 标签位置（在框上方）
        label_y = ymin - 10 if ymin - 10 > label_h else ymin + label_h
        cv2.rectangle(img, (xmin, label_y - label_h - 5), (xmin + label_w, label_y + 5), color, -1)

        # 绘制标签文字 (黑色)
        cv2.putText(img, label, (xmin, label_y - 5), font, font_scale, (0, 0, 0), thickness)

    return img


def draw_gt_image(img_path, labels_dir=None, class_names=None):
    """绘制带标注的 GT 图片"""
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # 如果有 YOLO TXT 标注，绘制 GT 框
    if labels_dir:
        stem = Path(img_path).stem
        label_path = Path(labels_dir) / f"{stem}.txt"
        if label_path.exists():
            h, w = img.shape[:2]
            bboxes = []
            try:
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])

                        # 转换为绝对坐标
                        xmin = int((x_center - bw / 2) * w)
                        ymin = int((y_center - bh / 2) * h)
                        xmax = int((x_center + bw / 2) * w)
                        ymax = int((y_center + bh / 2) * h)

                        bboxes.append((cls_id, xmin, ymin, xmax, ymax))

                img = draw_bboxes(img, bboxes, label_prefix="GT:")
            except Exception:
                pass

    # 转回 RGB 格式 (cv2 是 BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def draw_pred_image(img_path, predict_dir, image_name):
    """绘制带预测框的图片 (统一风格)"""
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # 尝试读取 YOLO 预测结果 TXT (在 labels/ 子目录下)
    stem = Path(image_name).stem
    pred_txt = Path(predict_dir) / "labels" / f"{stem}.txt"
    if pred_txt.exists():
        h, w = img.shape[:2]
        bboxes = []
        try:
            with open(pred_txt, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                    conf = float(parts[5])

                    xmin = int((x_center - bw / 2) * w)
                    ymin = int((y_center - bh / 2) * h)
                    xmax = int((x_center + bw / 2) * w)
                    ymax = int((y_center + bh / 2) * h)

                    bboxes.append((cls_id, xmin, ymin, xmax, ymax))

            img = draw_bboxes(img, bboxes, label_prefix="Pred:")
        except Exception:
            pass

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def display_comparison(predict_dir, gt_dir, labels_dir, image_index):
    """同步显示预测结果和 GT"""
    pairs, status = get_image_pairs(predict_dir, gt_dir)

    if not pairs:
        return None, None, status, gr.Slider(minimum=0, maximum=1, value=0, step=1)

    max_idx = len(pairs) - 1
    idx = int(image_index) % len(pairs)
    pair = pairs[idx]

    # 加载预测图片 (YOLO 输出已自带检测框)
    pred_array = np.fromfile(pair["predict"], dtype=np.uint8)
    pred_img = cv2.imdecode(pred_array, cv2.IMREAD_COLOR)
    if pred_img is not None:
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

    # 加载 GT 图片 (统一绘制风格)
    gt_img = draw_gt_image(pair["gt"], labels_dir)

    info = f"图片 {idx + 1}/{len(pairs)} | 文件: {pair['name']}"
    return pred_img, gt_img, info, gr.Slider(minimum=0, maximum=max_idx, value=idx, step=1)


def update_comparison_slider(predict_dir, gt_dir):
    """更新对比滑块"""
    pairs, status = get_image_pairs(predict_dir, gt_dir)
    if pairs:
        return gr.Slider(minimum=0, maximum=len(pairs) - 1, value=0, step=1)
    return gr.Slider(minimum=0, maximum=1, value=0, step=1)


# ==================== Gradio App ====================

def create_app():
    """创建 Gradio 应用"""
    with gr.Blocks(title="YOLOv8 训练结果可视化", theme=gr.themes.Soft()) as app:
        gr.Markdown("# YOLOv8 训练结果可视化")

        with gr.Tabs():
            # Tab 1: 训练指标
            with gr.Tab("训练指标"):
                gr.Markdown("### 训练 Loss / mAP / 混淆矩阵等指标")

                with gr.Row():
                    results_dir_input = gr.Textbox(
                        label="训练结果目录",
                        value=DEFAULT_RESULTS_DIR,
                        placeholder="runs/detect/luoshuan/train"
                    )
                    refresh_metrics_btn = gr.Button("刷新", variant="primary")

                metric_dropdown = gr.Dropdown(label="选择指标", choices=[])
                metric_image = gr.Image(label="指标图片", type="filepath")
                metric_status = gr.Textbox(label="状态", interactive=False)

                refresh_metrics_btn.click(
                    fn=update_metric_dropdown,
                    inputs=[results_dir_input],
                    outputs=[metric_dropdown]
                ).then(
                    fn=display_metric,
                    inputs=[results_dir_input, metric_dropdown],
                    outputs=[metric_image, metric_status]
                )

                metric_dropdown.change(
                    fn=display_metric,
                    inputs=[results_dir_input, metric_dropdown],
                    outputs=[metric_image, metric_status]
                )

            # Tab 2: 检测结果浏览
            with gr.Tab("检测结果"):
                gr.Markdown("### 模型推理结果图片浏览")

                with gr.Row():
                    predict_dir_input = gr.Textbox(
                        label="推理结果目录",
                        value=DEFAULT_PREDICT_DIR,
                        placeholder="runs/predict/luoshuan/infer"
                    )
                    refresh_detect_btn = gr.Button("刷新", variant="primary")

                detect_slider = gr.Slider(label="图片索引", minimum=0, maximum=1, value=0, step=1)
                detect_image = gr.Image(label="检测结果图片", type="filepath")
                detect_status = gr.Textbox(label="状态", interactive=False)

                refresh_detect_btn.click(
                    fn=update_detection_slider,
                    inputs=[predict_dir_input],
                    outputs=[detect_slider]
                ).then(
                    fn=display_detection,
                    inputs=[predict_dir_input, detect_slider],
                    outputs=[detect_image, detect_status, detect_slider]
                )

                detect_slider.change(
                    fn=display_detection,
                    inputs=[predict_dir_input, detect_slider],
                    outputs=[detect_image, detect_status, detect_slider]
                )

                with gr.Row():
                    prev_btn = gr.Button("上一张")
                    next_btn = gr.Button("下一张")

                prev_btn.click(
                    fn=lambda d, i: display_detection(d, max(0, int(i) - 1)),
                    inputs=[predict_dir_input, detect_slider],
                    outputs=[detect_image, detect_status, detect_slider]
                )

                next_btn.click(
                    fn=lambda d, i: display_detection(d, int(i) + 1),
                    inputs=[predict_dir_input, detect_slider],
                    outputs=[detect_image, detect_status, detect_slider]
                )

            # Tab 3: 预测 vs Ground Truth 对比
            with gr.Tab("预测 vs GT 对比"):
                gr.Markdown("### 预测结果与 Ground Truth 同步对比查看")

                with gr.Row():
                    with gr.Column():
                        predict_dir_cmp = gr.Textbox(
                            label="预测结果目录",
                            value=DEFAULT_PREDICT_DIR,
                            placeholder="runs/predict/luoshuan/infer"
                        )
                    with gr.Column():
                        gt_dir_cmp = gr.Textbox(
                            label="Ground Truth 图片目录",
                            value=DEFAULT_GT_DIR,
                            placeholder="法兰盘连接螺栓缺失"
                        )
                    with gr.Column():
                        labels_dir_cmp = gr.Textbox(
                            label="YOLO 标注目录",
                            value=DEFAULT_LABELS_DIR,
                            placeholder="data/luoshuan/labels/test"
                        )

                refresh_cmp_btn = gr.Button("加载对比数据", variant="primary")

                cmp_slider = gr.Slider(label="图片索引", minimum=0, maximum=1, value=0, step=1)
                cmp_status = gr.Textbox(label="状态", interactive=False)

                with gr.Row():
                    pred_image = gr.Image(label="预测结果", type="numpy")
                    gt_image = gr.Image(label="Ground Truth", type="numpy")

                with gr.Row():
                    prev_cmp_btn = gr.Button("上一张")
                    next_cmp_btn = gr.Button("下一张")

                refresh_cmp_btn.click(
                    fn=update_comparison_slider,
                    inputs=[predict_dir_cmp, gt_dir_cmp],
                    outputs=[cmp_slider]
                ).then(
                    fn=display_comparison,
                    inputs=[predict_dir_cmp, gt_dir_cmp, labels_dir_cmp, cmp_slider],
                    outputs=[pred_image, gt_image, cmp_status, cmp_slider]
                )

                cmp_slider.change(
                    fn=display_comparison,
                    inputs=[predict_dir_cmp, gt_dir_cmp, labels_dir_cmp, cmp_slider],
                    outputs=[pred_image, gt_image, cmp_status, cmp_slider]
                )

                prev_cmp_btn.click(
                    fn=lambda pd, gd, ld, i: display_comparison(pd, gd, ld, max(0, int(i) - 1)),
                    inputs=[predict_dir_cmp, gt_dir_cmp, labels_dir_cmp, cmp_slider],
                    outputs=[pred_image, gt_image, cmp_status, cmp_slider]
                )

                next_cmp_btn.click(
                    fn=lambda pd, gd, ld, i: display_comparison(pd, gd, ld, int(i) + 1),
                    inputs=[predict_dir_cmp, gt_dir_cmp, labels_dir_cmp, cmp_slider],
                    outputs=[pred_image, gt_image, cmp_status, cmp_slider]
                )

    return app


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 训练结果可视化 Web 应用")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="训练结果目录")
    parser.add_argument("--predict_dir", type=str, default=DEFAULT_PREDICT_DIR,
                        help="推理结果目录")
    parser.add_argument("--gt_dir", type=str, default=DEFAULT_GT_DIR,
                        help="Ground Truth 目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 设置默认值
    DEFAULT_RESULTS_DIR = args.results_dir
    DEFAULT_PREDICT_DIR = args.predict_dir
    DEFAULT_GT_DIR = args.gt_dir

    app = create_app()

    print("=" * 60)
    print("YOLOv8 训练结果可视化 Web 应用")
    print("=" * 60)
    print(f"\n启动地址: http://localhost:{args.port}")
    if args.share:
        print("公网链接将在启动后显示")
    print("\n按 Ctrl+C 停止服务")
    print("=" * 60)

    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        prevent_thread_lock=False,
        allowed_paths=["."],
    )
