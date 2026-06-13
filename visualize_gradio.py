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


def display_detection(predict_dir, image_index):
    """显示选定的检测结果图片"""
    if not predict_dir or not Path(predict_dir).exists():
        return None, "推理结果目录不存在", 0

    images = get_detection_images(predict_dir)
    if not images:
        return None, "未找到推理结果图片", 0

    idx = int(image_index) % len(images)
    return images[idx], f"图片 {idx + 1}/{len(images)}", len(images) - 1


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


def draw_gt_image(img_path, xml_dir=None):
    """绘制带标注的 GT 图片"""
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # 如果有 XML 标注，绘制 GT 框
    if xml_dir:
        stem = Path(img_path).stem
        xml_path = Path(xml_dir) / f"{stem}.xml"
        if xml_path.exists():
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    bndbox = obj.find("bndbox")
                    if bndbox is not None:
                        xmin = int(bndbox.find("xmin").text)
                        ymin = int(bndbox.find("ymin").text)
                        xmax = int(bndbox.find("xmax").text)
                        ymax = int(bndbox.find("ymax").text)
                        class_name = obj.find("name").text

                        # 绿色框表示 GT
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                        cv2.putText(img, f"GT: {class_name}", (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception:
                pass

    # 转回 RGB 格式 (cv2 是 BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def display_comparison(predict_dir, gt_dir, xml_dir, image_index):
    """同步显示预测结果和 GT"""
    pairs, status = get_image_pairs(predict_dir, gt_dir)

    if not pairs:
        return None, None, status, 0

    idx = int(image_index) % len(pairs)
    pair = pairs[idx]

    # 加载预测图片
    pred_array = np.fromfile(pair["predict"], dtype=np.uint8)
    pred_img = cv2.imdecode(pred_array, cv2.IMREAD_COLOR)
    if pred_img is not None:
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

    # 加载 GT 图片 (带标注)
    gt_img = draw_gt_image(pair["gt"], xml_dir)

    info = f"图片 {idx + 1}/{len(pairs)} | 文件: {pair['name']}"
    return pred_img, gt_img, info, len(pairs) - 1


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
                    fn=lambda d, i, mx: display_detection(d, (int(i) + 1) % (int(mx) + 1)),
                    inputs=[predict_dir_input, detect_slider, detect_slider],
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
                        xml_dir_cmp = gr.Textbox(
                            label="XML 标注目录 (可选)",
                            value=DEFAULT_GT_DIR,
                            placeholder="法兰盘连接螺栓缺失"
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
                    inputs=[predict_dir_cmp, gt_dir_cmp, xml_dir_cmp, cmp_slider],
                    outputs=[pred_image, gt_image, cmp_status, cmp_slider]
                )

                cmp_slider.change(
                    fn=display_comparison,
                    inputs=[predict_dir_cmp, gt_dir_cmp, xml_dir_cmp, cmp_slider],
                    outputs=[pred_image, gt_image, cmp_status, cmp_slider]
                )

                prev_cmp_btn.click(
                    fn=lambda pd, gd, xd, i: display_comparison(pd, gd, xd, max(0, int(i) - 1)),
                    inputs=[predict_dir_cmp, gt_dir_cmp, xml_dir_cmp, cmp_slider],
                    outputs=[pred_image, gt_image, cmp_status, cmp_slider]
                )

                next_cmp_btn.click(
                    fn=lambda pd, gd, xd, i, mx: display_comparison(pd, gd, xd, (int(i) + 1) % (int(mx) + 1)),
                    inputs=[predict_dir_cmp, gt_dir_cmp, xml_dir_cmp, cmp_slider, cmp_slider],
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
        prevent_thread_lock=True,
        allowed_paths=["."],
    )
