# YOLOv8 目标检测训练与测试 Pipeline

## 数据概览

| 项目 | 说明 |
|------|------|
| 数据格式 | VOC2007 XML + JPG |
| 类别数 | 2 |
| 类别名称 | `07010001`, `07010002` |
| 图片总数 | 300 张 |
| 标注方式 | 边界框 (bndbox: xmin, ymin, xmax, ymax) |

---

## 项目目录结构

```
dianwang/
├── plan.md                          # 本规划文档
├── requirements.txt                 # Python 依赖
├── convert_voc2yolo.py              # 1. VOC XML → YOLO TXT 格式转换脚本
├── split_dataset.py                 # 2. 数据集划分脚本 (train/val/test)
├── dataset.yaml                     # 3. YOLO 数据集配置文件
├── train.py                         # 4. 训练脚本
├── test.py                          # 5. 测试/推理脚本
├── visualize_results.py             # 6. 结果可视化脚本
├── data/                            # 数据处理后目录
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── runs/                            # 训练输出目录
│   └── detect/
└── weights/                         # 预训练权重 & 最佳模型
    └── best.pt
```

---

## 组件规划

### 1. 格式转换 (`convert_voc2yolo.py`)

**目标**: 将 VOC2007 XML 标注转换为 YOLO 格式 TXT 标注

**输入**: 
- 源目录: `挂点金具开口销缺失/*.xml` + `挂点金具开口销缺失/*.jpg`

**输出**: 
- `data/labels/` 下对应每个图片的 `.txt` 文件
- 图片复制到 `data/images/`

**YOLO 格式**: 每行 `class_id x_center y_center width height` (归一化到 [0,1])

**关键逻辑**:
- 解析 XML 提取 `<object>` 中的 `<name>` 和 `<bndbox>`
- 类别映射: `{"07010001": 0, "07010002": 1}`
- 坐标转换: `(xmin, ymin, xmax, ymax)` → `(x_center, y_center, w, h)` 归一化
- 处理无标注图片 (生成空 txt)

---

### 2. 数据集划分 (`split_dataset.py`)

**目标**: 将数据集划分为 train / val / test

**划分比例**: 
- train: 70% (210 张)
- val: 15% (45 张)
- test: 15% (45 张)

**策略**: 随机打乱后按比例划分，确保可复现 (固定 random seed)

---

### 3. 依赖文件 (`requirements.txt`)

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

### 4. YOLO 数据集配置 (`dataset.yaml`)

```yaml
path: ./data
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: "07010001"
  1: "07010002"
```

---

### 5. 训练脚本 (`train.py`)

**功能**:
- 加载 YOLOv8 预训练模型 (yolov8n.pt / yolov8s.pt)
- 读取 `dataset.yaml` 配置
- 执行训练，输出到 `runs/detect/`
- 保存最佳权重到 `weights/best.pt`

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| model | yolov8n.pt | 预训练模型 |
| epochs | 100 | 训练轮数 |
| batch | 16 | batch size |
| imgsz | 640 | 输入图像尺寸 |
| lr0 | 0.01 | 初始学习率 |
| patience | 50 | 早停轮数 |

---

### 6. 测试脚本 (`test.py`)

**功能**:
- 加载训练好的模型权重
- 在 test 集上评估 (mAP, precision, recall)
- 支持单张图片/文件夹推理
- 输出检测结果图

**模式**:
1. **评估模式**: 计算 test 集指标
2. **推理模式**: 对指定图片/文件夹进行检测，保存结果图

---

### 7. 结果可视化 (`visualize_results.py`)

**功能**:
- 读取训练日志，绘制 loss/mAP 曲线
- 展示测试集上的检测样例
- 生成混淆矩阵 (如可用)

---

## 执行流程

```bash
# Step 0: 安装依赖
pip install -r requirements.txt

# Step 1: 格式转换 (VOC XML → YOLO TXT)
python convert_voc2yolo.py

# Step 2: 数据集划分
python split_dataset.py

# Step 3: 训练
python train.py

# Step 4: 测试
python test.py --mode eval          # 评估模式
python test.py --mode infer --source data/images/test  # 推理模式

# Step 5: 可视化
python visualize_results.py
```

---

## 注意事项

1. **类别名称**: 当前使用数字编码 `07010001`/`07010002`，如有语义名称可在 `dataset.yaml` 中修改 `names` 字段
2. **GPU**: 训练需要 CUDA 环境，无 GPU 时自动使用 CPU (速度较慢)
3. **数据增强**: YOLOv8 内置 Mosaic、MixUp、HSV 等增强，可在 `train.py` 中调整
4. **模型选择**: 默认使用 yolov8n (nano)，可根据精度需求切换到 yolov8s/m/l/x
