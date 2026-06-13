# 大图分片训练与推理方案

## 问题分析

luoshuan 数据集中图片尺寸过大，经过 YOLOv8 的 640x640 缩放后目标占比极小，模型无法有效学习。

## 解决方案

在数据集划分阶段，先划分原图到 train/val/test，再根据划分好的原图生成分片子图数据集。推理时对大图分片预测并还原坐标。

### 关键设计

1. **分片尺寸**: `TILE_SIZE = 1000`
2. **重叠区域**: `OVERLAP = 200`
3. **小图不分割**: 宽高都 <= 1000 的图片直接复制
4. **convert_voc2yolo.py 不改动**，分片逻辑在 split 阶段完成

---

## 执行计划

### Phase 1: 数据集划分 + 分片（核心）

**修改文件**: `split_dataset.py`

**输出目录结构**:
```
data/luoshuan/
├── images/
│   ├── train/          # 原始图片（train集）
│   ├── val/            # 原始图片（val集）
│   ├── test/           # 原始图片（test集）
│   ├── train_cut/      # 分片子图（train集）
│   ├── val_cut/        # 分片子图（val集）
│   └── test_cut/       # 分片子图（test集）
├── labels/
│   ├── train/          # 原始标注（train集）
│   ├── val/            # 原始标注（val集）
│   ├── test/           # 原始标注（test集）
│   ├── train_cut/      # 子图标注（train集）
│   ├── val_cut/        # 子图标注（val集）
│   └── test_cut/       # 子图标注（test集）
└── tile_offsets/
    ├── train_cut.json  # train子图偏移信息
    ├── val_cut.json    # val子图偏移信息
    └── test_cut.json   # test子图偏移信息
```

**流程**:

1. 读取所有原始图片（`images/*.jpg`）和标注（`labels/*.txt`）
2. 随机划分为 train/val/test（90%/5%/5%）
3. 移动原始图片和标注到对应目录
4. **对每个子集（train/val/test）分别执行分片**：
   - 遍历该子集的原图（`images/{split}/*.jpg`）
   - 读取对应标注（`labels/{split}/*.txt`）
   - 判断是否需要分片（宽高 > 1000）
   - 小图：直接复制到 `_cut` 目录，标注也复制，偏移(0,0)
   - 大图：
     - 计算切分网格（含重叠）
     - 对每个子图：裁剪图片、筛选标注框（中心点在子图内）、转换相对坐标、保存
     - 记录偏移信息到 `tile_offsets/{split}_cut.json`

**标注坐标转换**:
```
原YOLO坐标（相对于原图）→ 子图YOLO坐标（相对于子图）
1. 反归一化得到原图绝对坐标: abs_x = yolo_x * orig_w
2. 减去偏移得到子图绝对坐标: sub_x = abs_x - offset_x
3. 重新归一化（相对于子图尺寸）: sub_yolo = sub_x / tile_w
```

**偏移信息格式** (`tile_offsets/train_cut.json`):
```json
{
  "原图名_tile_0_0.jpg": {
    "original": "原图名.jpg",
    "offset_x": 0,
    "offset_y": 0,
    "tile_w": 1000,
    "tile_h": 1000
  }
}
```

---

### Phase 2: 训练

**文件**: `train.py`（无需改动）

训练时使用 `dataset_luoshuan.yaml`，修改其路径指向 `_cut` 数据：
```yaml
path: ./data/luoshuan
train: images/train_cut
val: images/val_cut
test: images/test_cut
```

---

### Phase 3: 推理 - 分片预测与坐标还原

**修改文件**: `test.py`

**改动内容**:

1. 新增分片配置和函数：
   - `compute_tile_offsets()` - 分片偏移计算
   - `nms()` - 非极大值抑制去重
   - `tile_predict()` - 分片预测核心
   - `draw_predictions_from_boxes()` - 从合并框列表绘制结果

2. `tile_predict()` 逻辑：
   - 小图（<=1000）：直接预测
   - 大图：
     - 按相同规则切分子图（重叠）
     - 对每个子图调用模型预测
     - 子图预测坐标 + 偏移量 = 原图坐标
     - 按类别分别做 NMS 去重

3. 修改 `infer()` 函数：调用 `tile_predict()` 替代直接预测

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `convert_voc2yolo.py` | 不改动 | 保持原样 |
| `split_dataset.py` | 修改 | 新增分片逻辑，生成 _cut 子图数据集和偏移信息 |
| `dataset_luoshuan.yaml` | 修改 | 路径指向 _cut 数据 |
| `test.py` | 修改 | 新增分片推理和坐标还原 |

---

## 实现顺序

1. 实现 `split_dataset.py` 的分片逻辑
2. 修改 `dataset_luoshuan.yaml` 指向 _cut 数据
3. 训练模型
4. 实现 `test.py` 的分片推理逻辑
5. 端到端测试验证

---

## 注意事项

1. **标注框筛选**: 中心点在子图内的标注框才保留，跨边界的框裁剪到子图范围
2. **NMS 去重**: 重叠区域的目标会被多个子图检测到，需按类别分别 NMS
3. **偏移信息**: 必须保存，推理时用于坐标还原
4. **小图处理**: 宽高都 <= 1000 的图片直接复制，偏移(0,0)
5. **数据集隔离**: 先划分原图再分片，确保同一原图的子图不会跨数据集
