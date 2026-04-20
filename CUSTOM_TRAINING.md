# UniDet3D 自定义 RGB-D 数据训练指南

本文档覆盖从 **原始 RGB-D 数据 + labelme 标注** 到 **UniDet3D 训练与推理** 的完整流程，支持任意自定义类别和大批量数据扩展。

---

## 目录

1. [在新机器上配置环境](#1-在新机器上配置环境)
2. [日常使用：激活环境](#2-日常使用激活环境)
3. [数据采集与标注规范](#3-数据采集与标注规范)
4. [生成 UniDet3D 训练数据](#4-生成-unidet3d-训练数据)
5. [增加新类别](#5-增加新类别)
6. [训练配置与启动](#6-训练配置与启动)
7. [推理与可视化](#7-推理与可视化)
8. [批量数据扩展流程](#8-批量数据扩展流程)
9. [常见问题](#9-常见问题)
10. [端到端流程验证结果](#10-端到端流程验证结果5-帧示例数据)

---

## 1. 在新机器上配置环境

### 1.0 硬件与系统要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | NVIDIA，≥ 8 GB 显存 | RTX 3090 / 4070 及以上 |
| CUDA 驱动 | ≥ 12.1 | 12.1–12.4 |
| 系统 | Ubuntu 20.04 / 22.04 | Ubuntu 22.04 |
| 磁盘空间 | ≥ 20 GB（含 conda 环境） | ≥ 50 GB |
| GCC | 9–12（**不要用 GCC 13+**） | GCC 11 |

> **验证驱动版本**：`nvidia-smi` → 右上角 `CUDA Version: 12.x`

---

### 1.1 安装 Miniconda（已有 conda 跳过）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'source $HOME/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
conda config --set auto_activate_base false
```

---

### 1.2 克隆工程

```bash
git clone https://github.com/filaPro/unidet3d.git
cd unidet3d
# 或直接拷贝已有工程目录到目标机器
```

---

### 1.3 创建 conda 虚拟环境

```bash
conda create -n unidet3d python=3.10 -y
conda activate unidet3d
```

---

### 1.4 安装 PyTorch（CUDA 12.1）

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

> **CUDA 11.8 机器**：把 `cu121` 改为 `cu118`，后续 spconv 也用 `spconv-cu118`。

验证：
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 应输出: 2.1.2+cu121  True
```

---

### 1.5 安装 OpenMMLab 套件

```bash
pip install mmengine==0.9.0

# mmcv 需指定与 PyTorch + CUDA 匹配的预编译版本
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

pip install mmdet==3.3.0 mmsegmentation==1.2.0 mmpretrain==1.2.0

pip install mmdet3d==1.4.0
```

---

### 1.6 安装 spconv（稀疏卷积）

```bash
# CUDA 12.x 使用 cu120
pip install spconv-cu120==2.3.6

# CUDA 11.8 使用 cu118
# pip install spconv-cu118==2.3.6
```

---

### 1.7 安装 MinkowskiEngine（需编译）

MinkowskiEngine 没有预编译包，必须从源码编译，步骤较多，请按顺序执行：

**Step 1：安装编译依赖**

```bash
# openblas（MinkowskiEngine 的线性代数后端）
conda install -c conda-forge openblas -y

# 确保使用 GCC 11（Ubuntu 22.04 默认是 11，Ubuntu 20.04 可能需要手动安装）
sudo apt-get install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
gcc --version   # 应显示 gcc 11.x
```

**Step 2：设置编译环境变量**

```bash
# 找到 conda 环境中的 openblas 路径
CONDA_ENV_PATH=$(conda info --base)/envs/unidet3d
export OPENBLAS_PATH=$CONDA_ENV_PATH

# 避免 gfortran 链接问题
export CFLAGS="-I$CONDA_ENV_PATH/include"
export LDFLAGS="-L$CONDA_ENV_PATH/lib -Wl,-rpath,$CONDA_ENV_PATH/lib"
```

**Step 3：编译安装**

```bash
pip install ninja   # 加速编译

# 从 PyPI 安装（推荐，自动下载源码并编译）
pip install MinkowskiEngine==0.5.4 \
    --install-option="--blas=openblas" \
    --no-deps

# 如果上面失败，从 GitHub 源码编译：
# git clone https://github.com/NVIDIA/MinkowskiEngine.git
# cd MinkowskiEngine
# python setup.py install --blas=openblas --force_cuda
# cd ..
```

验证：
```bash
python -c "import MinkowskiEngine; print(MinkowskiEngine.__version__)"
# 应输出: 0.5.4
```

---

### 1.8 安装 torch-scatter

```bash
pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

---

### 1.9 安装其余依赖

```bash
pip install \
    transformers==4.40.2 \
    timm==0.9.16 \
    huggingface_hub==0.36.2 \
    accelerate==1.13.0 \
    open3d==0.17.0 \
    opencv-python==4.7.0.72 \
    scikit-image==0.25.2 \
    scikit-learn==1.2.2 \
    numpy==1.24.1 \
    scipy==1.15.3 \
    numba==0.57.0 \
    pandas==2.0.1 \
    matplotlib==3.5.2 \
    pycocotools==2.0.6 \
    nuscenes-devkit==1.1.10 \
    Shapely==1.8.5 \
    addict==2.4.0 \
    yapf==0.33.0 \
    termcolor==2.3.0 \
    rich==13.3.5 \
    tqdm==4.67.3 \
    plyfile==1.0.2 \
    trimesh==3.21.6 \
    plotly==5.18.0 \
    labelme
```

---

### 1.10 设置 conda 激活脚本（自动配置环境变量）

每次激活 `unidet3d` 环境时需要 `PYTHONPATH` 等变量生效，写入激活脚本避免每次手动 export：

```bash
ACTIVATE_DIR=$(conda info --base)/envs/unidet3d/etc/conda/activate.d
mkdir -p $ACTIVATE_DIR

cat > $ACTIVATE_DIR/unidet3d_env.sh << 'EOF'
#!/bin/bash
# UniDet3D environment variables
export PYTHONPATH=/path/to/unidet3d:$PYTHONPATH   # ← 改为实际工程路径
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF

chmod +x $ACTIVATE_DIR/unidet3d_env.sh
```

> **重要**：将 `/path/to/unidet3d` 替换为实际的工程根目录，例如 `/home/user/code/unidet3d/unidet3d`。

重新激活使其生效：
```bash
conda deactivate && conda activate unidet3d
```

---

### 1.11 验证完整安装

```bash
conda activate unidet3d
cd /path/to/unidet3d

python -c "
import torch, mmdet3d, spconv, MinkowskiEngine
import open3d, transformers, cv2
print('PyTorch:          ', torch.__version__)
print('CUDA available:   ', torch.cuda.is_available())
print('GPU:              ', torch.cuda.get_device_name(0))
print('mmdet3d:          ', mmdet3d.__version__)
print('spconv:           ', spconv.__version__)
print('MinkowskiEngine:  ', MinkowskiEngine.__version__)
print('open3d:           ', open3d.__version__)
print('transformers:     ', transformers.__version__)
print('All imports OK!')
"
```

预期输出：
```
PyTorch:           2.1.2+cu121
CUDA available:    True
GPU:               NVIDIA GeForce RTX XXXX
mmdet3d:           1.4.0
spconv:            2.3.6
MinkowskiEngine:   0.5.4
open3d:            0.17.0
transformers:      4.40.2
All imports OK!
```

---

### 1.12 下载 Depth Anything V2 模型权重（可选，用于数据生成）

```bash
# 方法一：通过 huggingface_hub 自动下载（首次运行 prepare_custom_rgbd.py 时自动触发）
# 模型会缓存到 ~/.cache/huggingface/

# 方法二：手动下载（离线环境）
mkdir -p work_dirs/tmp
# 下载 depth-anything-v2-large（~1.3 GB）
# 从 https://huggingface.co/depth-anything/Depth-Anything-V2-Large 下载 pytorch_model.bin
# 放置到 ~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Large/
```

---

## 2. 日常使用：激活环境

```bash
conda activate unidet3d
cd /path/to/unidet3d   # 进入工程目录
```

激活后 `PYTHONPATH`、`CUDA_HOME` 等变量由激活脚本自动设置。

---

## 3. 数据采集与标注规范

### 3.1 目录结构

每批数据需组织成如下结构：

```
/path/to/dataset/
├── camera_intrinsics.json      # 相机内参（一次写入，所有帧共用）
├── depth/
│   └── depth_<ID>.npy          # 深度图，float32，单位 mm，形状 (H,W)
└── images/
    ├── rgb_<ID>.jpg             # 彩色图，与深度图像素对齐
    └── rgb_<ID>.json            # labelme polygon 标注文件（同名）
```

`<ID>` 为任意字符串，rgb 与 depth 的 `<ID>` 必须一一对应。

### 3.2 相机内参格式

```json
{
  "fx": 366.9,  "fy": 366.9,
  "cx": 318.6,  "cy": 242.5,
  "width": 640, "height": 480
}
```

### 3.3 labelme 标注规范

使用 [labelme](https://github.com/labelmeai/labelme) 标注工具，标注规则：

| 要求 | 说明 |
|------|------|
| 标注工具 | labelme（polygon 模式） |
| 标注形状 | polygon（多边形，紧贴物体轮廓） |
| 类别名称 | 与训练 config 中 `classes_custom` 完全一致，区分大小写 |
| 背景 | 不需要标注，无 polygon 的区域自动视为背景 |
| 深度建议 | 8m 以内的物体标注效果最佳，超过 8m 的实例会被自动过滤 |

**标注质量建议：**
- 多边形尽量贴紧物体边缘（避免包含大量背景像素）
- 对于被遮挡的物体，只标注可见部分
- 每个物体实例单独标一个 polygon，不要合并

### 3.4 深度数据质量

| 距离 | 深度质量 | 3D 框质量 |
|------|---------|----------|
| 0–3m | 优秀 | 高精度 AABB |
| 3–6m | 良好 | 较好，DA2 补全后准确 |
| 6–8m | 一般 | DA2 辅助 + 类别先验兜底 |
| >8m  | 差 | 自动跳过，不生成训练框 |

---

## 4. 生成 UniDet3D 训练数据

### 4.1 运行数据生成脚本

```bash
cd /home/rossi/code/python/unidet3d/unidet3d

python tools/prepare_custom_rgbd.py \
    --data-dir  /path/to/dataset       \  # 数据目录（含 depth/ images/ camera_intrinsics.json）
    --out-dir   data/custom            \  # 输出训练数据目录
    --vis-dir   results/custom_vis     \  # 可视化输出目录
    --val-ratio 0.2                    \  # 验证集比例（最后 20% 帧）
    --erode-px  2                      \  # mask 腐蚀半径（去边缘飞点）
    --device    cuda:0                    # Depth Anything V2 运行设备
```

**可选参数：**
- `--no-da2`：跳过 Depth Anything V2，只用传感器深度（速度快但近处小物体可能效果差）

### 4.2 输出说明

```
data/custom/
├── points/            # 每帧点云：float32 (N,6) [x,y,z,R,G,B]
├── instance_mask/     # 实例 mask：int64  (N,)  实例 ID（-1=背景）
├── semantic_mask/     # 语义 mask：int64  (N,)  类别 ID（-1=背景）
├── super_points/      # 超点 mask：int64  (N,)  SLIC 超点 ID
├── custom_infos_train.pkl  # 训练集标注
└── custom_infos_val.pkl    # 验证集标注

results/custom_vis/
├── scene_XXXX_2d.jpg       # RGB 图 + 投影 3D 框（质量校验）
├── scene_XXXX_3d.png       # Open3D 3D 点云 + 框可视化
└── scene_XXXX_depth.jpg    # 传感器深度 vs 融合深度对比
```

### 4.3 深度增强原理（三级策略）

```
输入: 传感器深度(稀疏) + RGB
        ↓
[级别1] Depth Anything V2（DA2）估计稠密相对深度
        + 最小二乘标定到 metric 尺度（利用传感器有效像素）
        + 加权融合（近处信任传感器，远处信任 DA2）
        ↓
[级别2] 飞点清理
        mask 腐蚀（去边缘） → 统计离群点过滤 → DBSCAN 取最大簇 → 百分位裁剪
        ↓
[级别3] 类别先验兜底
        深度覆盖率 < 25% 时，用多边形投影尺寸 + 类别典型尺寸生成框
        ↓
输出: 3D AABB [cx, cy, cz, dx, dy, dz]（米，相机坐标系）
```

### 4.4 查看可视化结果

生成后检查 `results/custom_vis/` 中的 `*_2d.jpg`，确认：
- [ ] 3D 框投影到图像上位置正确
- [ ] 标注文字中的尺寸合理（car: ~1.8×1.5×4m，chair: ~0.6×0.9×0.6m，person: ~0.5×1.7×0.4m）
- [ ] 标签 `[depth(XX%)]` 覆盖率正常，`[prior(XX%)]` 说明深度稀疏用了先验

---

## 5. 增加新类别

### 5.1 修改数据生成脚本

编辑 `tools/prepare_custom_rgbd.py`，找到以下两处并修改：

```python
# 1. 类别列表（顺序即为类别 ID：0, 1, 2, ...）
CLASS_NAMES = ['car', 'chair', 'person', 'table', 'door']  # ← 追加新类别

# 2. 类别先验尺寸 [dx(横向), dy(垂直), dz(深度方向)] 单位:米
CLASS_PRIORS = {
    'car':    np.array([1.8, 1.5, 4.2]),
    'chair':  np.array([0.6, 0.9, 0.6]),
    'person': np.array([0.5, 1.7, 0.4]),
    'table':  np.array([1.2, 0.8, 0.8]),   # ← 追加新类别先验
    'door':   np.array([0.1, 2.1, 1.0]),
}

# 3. 可视化颜色（可选）
CLASS_COLORS_RGB = {
    ...
    'table':  np.array([200, 100, 50], dtype=np.float64) / 255.,
    'door':   np.array([100, 200, 150], dtype=np.float64) / 255.,
}
```

### 5.2 修改训练配置

编辑 `configs/unidet3d_custom.py`，修改第一行类别列表：

```python
classes_custom = ['car', 'chair', 'person', 'table', 'door']  # ← 与脚本保持一致
```

> ⚠️ **注意**：类别列表的顺序必须与 `CLASS_NAMES` 完全一致，且新增类别必须重新生成全部 PKL 和 .bin 文件。

### 5.3 labelme 标注中使用新类别名

在 labelme 中添加 polygon 时，label 字段填入新类别名（与 `CLASS_NAMES` 一致）。

---

## 6. 训练配置与启动

### 6.1 训练配置文件

配置文件位于 `configs/unidet3d_custom.py`，主要可调参数：

```python
# ── 类别 ──────────────────────────────────────────────────
classes_custom = ['car', 'chair', 'person']   # 修改为实际类别

# ── 模型超参 ──────────────────────────────────────────────
num_channels = 32      # backbone 通道数，增大→精度↑内存↑
voxel_size   = 0.02    # 体素大小(m)，减小→分辨率↑速度↓

# ── 训练超参 ──────────────────────────────────────────────
optim_wrapper = dict(
    optimizer=dict(lr=0.0001))   # 学习率，finetune 时可用 1e-5

train_cfg = dict(
    max_epochs=512,              # 总 epoch 数
    val_interval=16)             # 每 N epoch 验证一次

# ── 可选：从预训练模型加载 backbone ──────────────────────
# load_from = 'work_dirs/tmp/unidet3d.pth'
```

### 6.2 启动训练

**从头训练（推荐数据量 > 50 场景）：**
```bash
conda activate unidet3d
cd /home/rossi/code/python/unidet3d/unidet3d

PYTHONPATH=. python tools/train.py configs/unidet3d_custom.py \
    --work-dir work_dirs/custom_exp1
```

**微调预训练 backbone（推荐数据量 < 50 场景）：**
```bash
# 先取消注释 config 末尾的 load_from，然后：
PYTHONPATH=. python tools/train.py configs/unidet3d_custom.py \
    --work-dir work_dirs/custom_finetune \
    --cfg-options load_from=work_dirs/tmp/unidet3d.pth
```

**调整 batch size（根据显存）：**
```bash
PYTHONPATH=. python tools/train.py configs/unidet3d_custom.py \
    --cfg-options train_dataloader.batch_size=2
```

### 6.3 监控训练

```bash
# 查看 loss 曲线
tail -f work_dirs/custom_exp1/TIMESTAMP/vis_data/scalars.json

# 或用 tensorboard
tensorboard --logdir work_dirs/custom_exp1
```

训练 log 示例：
```
Epoch(train) [16][4/4]  lr: 9.97e-05  loss: 8.32  det_loss: 8.32
```

---

## 7. 推理与可视化

### 7.1 单帧 RGB-D 推理

```bash
conda activate unidet3d
cd /home/rossi/code/python/unidet3d/unidet3d

PYTHONPATH=. python tools/infer_rgbd.py \
    --data-dir   /path/to/dataset                   \
    --checkpoint work_dirs/unidet3d_custom/epoch_64.pth \
    --config     configs/unidet3d_custom.py          \
    --out-dir    results/custom_infer                \
    --score-thr  0.3                                 \
    --frame-idx  0                                   \
    --dataset-tag custom                             \
    --classes    car chair person
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--dataset-tag` | 必须与 config 中 `decoder.datasets` 一致（自定义数据集填 `custom`） |
| `--classes` | 类别名列表，与训练 config 中 `classes_custom` 顺序一致 |
| `--score-thr` | 置信度阈值，训练样本少时建议调低至 0.1–0.2 |
| `--frame-idx` | 帧编号，0 为第一帧 |

数据目录支持以下两种结构（自动识别）：
```
# 结构 A（旧格式）          # 结构 B（新格式）
dataset/                  dataset/
├── rgb_*.jpg             ├── images/rgb_*.jpg
├── depth_*.npy           ├── depth/depth_*.npy
└── camera_intrinsics.json└── camera_intrinsics.json
```

### 7.2 推理结果

```
results/custom_infer/
├── <frame_name>_2d.jpg          # RGB 图上的 2D 投影框
├── <frame_name>_3d.png          # 3D 点云 + 框（matplotlib）
└── <frame_name>_detections.txt  # 文本结果：class, score, cx, cy, cz, dx, dy, dz
```

检测结果格式示例（实测）：
```
class, score, cx, cy, cz, dx, dy, dz
car, 0.417, -1.65, -0.85, 1.28, 1.66, 2.35, 2.67
car, 0.348, -0.59,  0.33, 2.47, 2.30, 1.72, 1.33
chair, 0.197, -0.93, 0.38, 1.07, 0.53, 1.16, 1.54
```

### 7.3 使用 6 数据集预训练模型推理（ScanNet 类别）

如果使用官方预训练权重（18 类 ScanNet 词表）：

```bash
PYTHONPATH=. python tools/infer_rgbd.py \
    --data-dir   /path/to/dataset \
    --checkpoint work_dirs/tmp/unidet3d.pth \
    --config     configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py \
    --out-dir    results/scannet_infer \
    --dataset-tag scannet
    # 不加 --classes 时默认使用 SCANNET_CLASSES（18 类）
```

---

## 8. 批量数据扩展流程

当采集了新的一批数据时，按以下步骤操作：

### Step 1：整理数据目录

```
/path/to/new_batch/
├── camera_intrinsics.json
├── depth/
│   └── depth_<ID>.npy  (多个文件)
└── images/
    ├── rgb_<ID>.jpg
    └── rgb_<ID>.json   ← labelme 标注
```

### Step 2：生成训练数据

```bash
python tools/prepare_custom_rgbd.py \
    --data-dir /path/to/new_batch \
    --out-dir  data/custom_new    \
    --vis-dir  results/new_batch_vis
```

### Step 3：检查可视化，确认 3D 框质量

```bash
# 在 VSCode 中直接查看 results/new_batch_vis/*.jpg
# 重点检查：
#   - 2d.jpg 中框的位置是否正确
#   - 标注尺寸是否合理（见第 3.4 节标准）
```

### Step 4：合并多批数据（可选）

```bash
python tools/merge_custom_data.py \
    --src data/custom data/custom_new \
    --dst data/custom_merged          \
    --val-ratio 0.15
```

> 注：`tools/merge_custom_data.py` 的功能：合并多个 `data/*/` 目录的 `.bin` 文件和 PKL 文件。

### Step 5：重新训练或继续训练

```bash
# 继续上次训练（resume）
PYTHONPATH=. python tools/train.py configs/unidet3d_custom.py \
    --work-dir work_dirs/custom_exp2 \
    --resume   work_dirs/custom_exp1/latest.pth

# 或从头训练新数据
PYTHONPATH=. python tools/train.py configs/unidet3d_custom.py \
    --work-dir work_dirs/custom_exp2 \
    --cfg-options train_dataloader.dataset.data_root=data/custom_merged/
```

---

## 9. 常见问题

### Q1：训练 loss 不下降

- 数据量太少（< 20 场景）时正常，建议微调预训练模型
- 检查 3D 框质量：用 `results/*_2d.jpg` 确认框正确
- 尝试降低学习率：`optimizer.lr=0.00005`

### Q2：某些实例被跳过（"有效点 < 10"）

- 该实例在深度图中几乎没有有效深度值
- 对策：确保物体在相机有效量程内（0.2–8m），或去掉该标注

### Q3：3D 框尺寸明显偏大

- 检查深度数据是否有大量飞点
- 增大腐蚀参数：`--erode-px 4`
- 深度覆盖率低的实例会显示 `[prior(XX%)]`，说明使用了类别先验

### Q4：推理时类别全为 "otherfurniture"

- 推理时使用的 checkpoint 是 6 数据集预训练模型（ScanNet 词表）
- 必须使用自定义训练的 checkpoint 和对应的 config

### Q5：想用预训练权重加速训练

```python
# configs/unidet3d_custom.py 末尾取消注释：
load_from = 'work_dirs/tmp/unidet3d.pth'
```
backbone 权重会被加载，decoder（类别头）重新初始化。

### Q6：多 GPU 训练

```bash
PYTHONPATH=. python -m torch.distributed.launch \
    --nproc_per_node=4 tools/train.py configs/unidet3d_custom.py \
    --launcher pytorch
```

---

## 10. 端到端流程验证结果（5 帧示例数据）

以下为使用 5 帧示例数据（`/home/rossi/dataset/private/sampled_data`）完成完整流程的实测记录。

### 10.1 数据规模

| 项目 | 数值 |
|------|------|
| 原始 RGB 帧数 | 5 |
| 标注类别 | car, chair, person |
| 训练集场景数 | 4 |
| 验证集场景数 | 1 |
| 每场景平均点数 | ~80,000 |

### 10.2 训练结果

训练从随机初始化开始，运行 64 epoch：

```
Epoch  2:  loss = 10.49  (起始)
Epoch  8:  loss =  9.38
Epoch 16:  loss =  8.98  (首次验证: car_AP_0.25 = 0.005)
Epoch 32:  loss =  8.37  (验证: car_AP_0.25 = 0.000)
Epoch 48:  loss =  7.96  (验证: car_AP_0.25 = 0.000)
Epoch 64:  loss =  7.34  (验证: car_AP_0.25 = 0.036)
```

> **说明**：5 帧数据量极少，mAP 偏低属正常。实际使用建议准备 ≥ 50 帧（各类别均匀分布）。
> 预训练 backbone 微调（取消注释 `load_from`）可显著加速收敛。

### 10.3 推理示例

对第 0 帧（汽车展厅场景）的推理结果：
- 共产生 72 个候选框，20 个高于阈值 0.1
- 最高置信度检测：`car 0.417` at `(-1.65, -0.85, 1.28)m`，尺寸 `1.66×2.35×2.67m`
- 检测结果图：`results/custom_infer/rgb_d2c_..._2d.jpg`

### 10.4 checkpoint 路径

```
work_dirs/unidet3d_custom/
├── epoch_32.pth    # 中期 checkpoint
├── epoch_48.pth
└── epoch_64.pth    # 最终 checkpoint（推荐用于推理）
```

---

## 附录：关键文件路径

| 文件 | 说明 |
|------|------|
| `tools/prepare_custom_rgbd.py` | RGB-D + labelme → UniDet3D 训练数据（含 DA2 深度增强） |
| `tools/infer_rgbd.py` | 单帧 RGB-D 推理 + 可视化 |
| `configs/unidet3d_custom.py` | 自定义数据集训练配置 |
| `unidet3d/scannet_dataset.py` | `CustomRGBDDetDataset`：支持任意类别的数据集类 |
| `data/custom/` | 训练数据（.bin 文件 + .pkl 标注） |
| `work_dirs/tmp/unidet3d.pth` | 6 数据集预训练权重（可用于 backbone 初始化） |
| `results/custom_vis/` | 数据生成可视化结果 |