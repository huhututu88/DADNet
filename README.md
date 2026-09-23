# DADNet

**面向工业表面缺陷分割的双注意力 U-Net**

基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.2.2 实现，在 NEU-Seg 热轧带钢缺陷数据集上验证。

<!--
  放好结构图后，把下面这段的注释去掉即可（图存到 resources/dadnet_arch.png）。
  图片不存在时先保持注释状态，否则 GitHub 上会显示一个坏掉的图片图标。

<p align="center">
  <img src="resources/dadnet_arch.png" alt="DADNet architecture" width="90%">
</p>
-->

---

## 目录

- [0. 简介](#0-简介)
- [1. 环境安装](#1-环境安装)
- [2. 数据集准备](#2-数据集准备)
- [3. 训练](#3-训练)
- [4. 测试与评估](#4-测试与评估)
- [5. 消融实验](#5-消融实验)
- [6. 可视化](#6-可视化)
- [7. 常见问题](#7-常见问题)
- [8. 引用](#8-引用)
- [9. 致谢与许可](#9-致谢与许可)

---

## 0. 简介

DADNet 是一个用于**工业表面缺陷分割**的编码器-解码器网络。它把标准 U-Net 的
两个关键环节替换为本文提出的模块：

| 模块 | 位置 | 作用 |
|---|---|---|
| **DFWE**<br>Dual-branch Feature Weighted Encoding | 编码器下采样块 | 将特征按通道一分为二，用两组独立卷积分别提取后经 `Reconstruct` 重建，抑制背景纹理、突出缺陷响应 |
| **MDCF**<br>Multi-scale Difference Convolution Fusion | 解码器跳连融合 | 用共享分支提取公共分量，再对两个残差分量施加不同尺度的深度可分离卷积，兼顾大缺陷区域与小划痕 |

此外编码器使用 **ECA** 通道注意力，下采样使用**平均池化**（论文默认）。

两个模块都可以通过配置开关关闭，方便复现消融实验，见 [第 5 节](#5-消融实验)。

**模型规格**（`feature_scale=2`，输入 3×256×256）：

| 指标 | 数值 |
|---|---|
| 参数量 | 2.06 M |
| 首层通道数 | 32（64/128/256/512/1024 ÷ 2） |

---

## 1. 环境安装

### 1.1 硬件与驱动

开发环境：Ubuntu，4 × NVIDIA RTX 4090 (24 GB)，单卡即可训练全部配置。

### 1.2 已验证的环境组合

| 组件 | 版本 |
|---|---|
| Python | 3.10 / 3.11 |
| PyTorch | 2.1.1 + CUDA 11.8 |
| torchvision | 0.16.1 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |
| PyWavelets | 1.8.0 |
| einops | 0.8.1 |

> `PyWavelets` 和 `einops` 是 DADNet 额外引入的依赖，**不在** MMSegmentation 的
> requirements 里。用 `pip install -r requirements.txt` 会自动装好。

### 1.3 安装步骤

```bash
conda create -n dadnet python=3.10 -y
conda activate dadnet

# 1) 按你的 CUDA 版本装 PyTorch（示例为 CUDA 11.8）
pip install torch==2.1.1 torchvision==0.16.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 2) 装 MMCV（必须用 openmim 装带 CUDA 算子的版本）
pip install -U openmim
mim install "mmcv>=2.0.0rc4,<2.2.0"

# 3) 装 DADNet
cd DADNet
pip install -v -e .

# 4) 验证安装
python -c "import mmseg, pywt, einops; print(mmseg.__version__)"
```

安装成功后应输出 `1.2.2`。

<details>
<summary>不用 conda 的纯 pip 安装</summary>

```bash
pip install -r requirements.txt
pip install -v -e .
```
</details>

### 1.4 常见安装问题

| 报错 | 原因与解决 |
|---|---|
| `No module named 'pywt'` | 缺 PyWavelets：`pip install PyWavelets` |
| `No module named 'einops'` | `pip install einops` |
| `mmcv._ext` 缺失 / `No module named 'mmcv.ops'` | mmcv 装成了纯 Python 版。卸载后用 `mim install mmcv==2.1.0` 重装 |
| `undefined symbol` | PyTorch 与 mmcv 的 CUDA 版本不匹配，重装二者 |
| `EncoderDecoder is not in the mmengine::model registry` | 配置里缺 `default_scope = 'mmseg'`，见 `configs/dadnet/_base_/default_runtime.py` |

---

## 2. 数据集准备

### 2.1 使用的数据集

| 名称 | 类别数 | 图像后缀 | 掩码后缀 | 说明 |
|---|---|---|---|---|
| **NEU-Seg** | 4 | `.bmp` | `.png` | 热轧带钢表面缺陷：background, In, Pa, Sc |

请从数据集官方渠道下载原始数据，并遵守其许可与引用要求。

### 2.2 目录结构

准备好后，数据集目录应长这样：

```
data/NEU-Seg/
├── img_dir/
│   ├── train/*.bmp
│   ├── val/*.bmp
│   └── test/*.bmp
├── ann_dir/
│   ├── train/*.png
│   ├── val/*.png
│   └── test/*.png
├── train.json
├── val.json
└── test.json
```

其中 `*.json` 是**数据集索引文件**，由 `tools/prepare_dataset.py` 生成。
`SD900Dataset` 从这些 JSON 读取图像-掩码配对，而不是扫描目录。

### 2.3 索引文件格式

JSONL（每行一个 JSON 对象），两个字段：

```json
{"image_path": "img_dir/train/defect_001.bmp", "seg_mask_path": "ann_dir/train/defect_001.png"}
```

路径可以是**相对于 `data_root` 的相对路径**（推荐，数据集可随意搬移），
也可以是绝对路径。图像与掩码**按文件名主干（stem）配对**。

### 2.4 生成索引文件

**情况 A：原始数据已按 train/val/test 分好目录**

```bash
python tools/prepare_dataset.py --data-root data/NEU-Seg \
    --splits train val test
```

**情况 B：原始数据是一个大目录，需要自己划分**

```bash
# 先预览数量，不写任何文件
python tools/prepare_dataset.py --data-root data/NEU-Seg \
    --split-ratio 0.7 0.15 0.15 --seed 0 --dry-run

# 确认无误后实际划分，并把文件移动到 img_dir/<split>/ 与 ann_dir/<split>/
python tools/prepare_dataset.py --data-root data/NEU-Seg \
    --split-ratio 0.7 0.15 0.15 --seed 0 --move
```

脚本会自动校验图像与掩码一一对应，缺失配对时直接报错并列出示例文件名。

### 2.5 复用同一个数据集给多个项目

把数据集放在任意位置，在配置里改 `data_root` 即可。若不想改配置，
可以建软链接：

```bash
ln -s /your/data/NEU-Seg data/NEU-Seg
```

---

## 3. 训练

### 3.1 单卡训练

```bash
python tools/train.py configs/dadnet/dadnet_unet32_neu_seg.py
```

日志与权重保存到 `work_dirs/dadnet_unet32_neu_seg/`。

指定 GPU 与工作目录：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/dadnet/dadnet_unet32_neu_seg.py \
    --work-dir work_dirs/my_run
```

### 3.2 多卡训练

```bash
bash tools/dist_train.sh configs/dadnet/dadnet_unet32_neu_seg.py 4
```

> 多卡时请按卡数等比例放大 `train_dataloader.batch_size`，并相应调整学习率。

### 3.3 混合精度

```bash
python tools/train.py configs/dadnet/dadnet_unet32_neu_seg.py --amp
```

### 3.4 断点续训

```bash
python tools/train.py configs/dadnet/dadnet_unet32_neu_seg.py --resume
```

### 3.5 从预训练权重微调

本仓库不附带预训练权重。若你有自己之前训练好的 checkpoint（例如在别的数据集上
训过的 DADNet），可以直接用它初始化：

```bash
python tools/train.py configs/dadnet/dadnet_unet32_neu_seg.py \
    --cfg-options load_from=/path/to/your_checkpoint.pth
```

`load_from` 只加载权重，不恢复优化器状态和迭代数；要接着上次训完用 `--resume`。

### 3.6 各配置的训练超参

| 配置 | 数据集 | lr (Adam) | iters | batch | 输入尺寸 |
|---|---|---|---|---|---|
| `dadnet_unet32_neu_seg.py` | NEU-Seg | 1e-4 | 10 000 | 8 | 256×256 |

学习率使用 PolyLR（power 0.9，`eta_min=1e-5`），权值衰减 1e-5，随机种子 0，
每 100 个 iteration 验证一次并保存最优 mIoU 权重。

### 3.7 数据增强

训练管线固定在 `configs/dadnet/_base_/datasets/*.py` 中的 `train_pipeline`：

1. `LoadImageFromFile` → `LoadAnnotations`
2. `Resize` 到 256×256（保持长宽比）
3. `Random_Choices`（随机旋转 + 随机翻转的组合）
4. `PhotoMetricDistortion`（亮度/对比度/饱和度/色调扰动）
5. `PackSegInputs`

想换增强策略时改这里即可。

---

## 4. 测试与评估

```bash
python tools/test.py configs/dadnet/dadnet_unet32_neu_seg.py \
    work_dirs/dadnet_unet32_neu_seg/best_mIoU_iter_XXXX.pth
```

输出逐类 IoU / Acc / Dice / Fscore / Precision / Recall，以及总体 mIoU、mDice、mFscore。

### 4.1 常用选项

```bash
# 把预测结果保存为图片，便于挑 case 分析
python tools/test.py <config> <checkpoint> --show-dir vis_pred

# 把指标写入 json
python tools/test.py <config> <checkpoint> --work-dir results/

# 测试时增强
python tools/test.py <config> <checkpoint> --tta
```

### 4.2 在验证集上评估

把配置里的 `test_dataloader` 指向 `val.json` 即可：

```bash
python tools/test.py configs/dadnet/dadnet_unet32_neu_seg.py <ckpt> \
    --cfg-options test_dataloader.dataset.data_prefix.img_path=val.json \
                  test_dataloader.dataset.data_prefix.seg_map_path=val.json
```

### 4.3 打印参数量

```bash
python -c "
import torch, mmseg.models
from mmengine.config import Config
from mmseg.registry import MODELS
from mmengine.registry import DefaultScope
c = Config.fromfile('configs/dadnet/dadnet_unet32_neu_seg.py')
with DefaultScope.overwrite_default_scope('mmseg'):
    m = MODELS.build(c.model)
print('%.2f M' % (sum(p.numel() for p in m.parameters()) / 1e6))
"
```

---

## 5. 消融实验

DFWE 与 MDCF 都可以通过配置开关关闭。**不需要改任何源码**：

```bash
# w/o DFWE
python tools/train.py configs/dadnet/dadnet_unet32_neu_seg.py \
    --cfg-options model.backbone.is_catch=False \
    --work-dir work_dirs/ablation_wo_dfwe

# w/o MDCF
python tools/train.py configs/dadnet/dadnet_unet32_neu_seg.py \
    --cfg-options model.backbone.is_fusion=False \
    --work-dir work_dirs/ablation_wo_mdcf

# MDCF 卷积核换成 9x9
python tools/train.py configs/dadnet/dadnet_unet32_neu_seg.py \
    --cfg-options model.backbone.kernel_sizes="(9,9)" \
    --work-dir work_dirs/ablation_k9

# 池化方式换成最大池化
python tools/train.py configs/dadnet/dadnet_unet32_neu_seg.py \
    --cfg-options model.backbone.pool_type=max \
    --work-dir work_dirs/ablation_maxpool
```

可用的骨干开关：

| 参数 | 取值 | 默认 | 含义 |
|---|---|---|---|
| `is_catch` | `True` / `False` | `True` | 是否启用 DFWE |
| `is_fusion` | `True` / `False` | `True` | 是否启用 MDCF |
| `kernel_sizes` | 任意二元组 | `(5, 7)` | MDCF 两支深度卷积核尺寸 |
| `pool_type` | `'avg'` / `'max'` | `'avg'` | 下采样池化方式 |

> 注意 `kernel_sizes` 这类元组在 `--cfg-options` 里要写成带引号的字符串
> `"(9,9)"`，否则会被解析成两个参数。

---

## 6. 可视化

### 6.1 TensorBoard

默认只存本地日志（`LocalVisBackend`）。要同时写 TensorBoard，先装 tensorboard：

```bash
pip install tensorboard
```

然后编辑 `configs/dadnet/_base_/default_runtime.py`，把 `vis_backends` 那一行改成：

```python
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
```

再正常训练，最后启动：

```bash
tensorboard --logdir work_dirs --port 6006
```

> 这一步不能靠 `--cfg-options` 传，因为 `dict(type='...')` 不是合法的 Python 字面量，
> 会被当成字符串传进去，`visualizer` 构建时报错。改配置文件最省事。

### 6.2 从日志画曲线

需要额外装 seaborn（matplotlib 是核心依赖，已随 `requirements.txt` 装好）：

```bash
pip install seaborn

python tools/analysis_tools/analyze_logs.py \
    work_dirs/dadnet_unet32_neu_seg/*/vis_data/scalars.json \
    --keys mIoU mDice --legend dadnet --out curve.png
```

`analyze_logs.py` 没有子命令，日志路径直接作为位置参数传入。

---

## 7. 常见问题

<details>
<summary><b>报 <code>KeyError: 'EncoderDecoder is not in the mmengine::model registry'</code></b></summary>

配置缺少 `default_scope = 'mmseg'`。本项目所有配置都继承了
`configs/dadnet/_base_/default_runtime.py`，它设置了该项。自定义配置时记得一并继承。
</details>

<details>
<summary><b>报找不到文件 / 数据集路径相关的错（<code>FileNotFoundError</code>、<code>AssertionError</code>）</b></summary>

多半是这两个原因之一：

1. `data_root` 指错了。配置里的 `data_root` 是相对**你运行命令时所在的目录**解析的
   （不是相对配置文件），所以在项目根目录下跑最省事。
2. 还没跑 `tools/prepare_dataset.py` 生成 `{train,val,test}.json`。
   先用 `--dry-run` 确认配对数量正常，再去掉 `--dry-run` 实际生成。
</details>

<details>
<summary><b>想换输入分辨率怎么办？</b></summary>

同时改三处：`model.data_preprocessor.size`、各 dataloader 的 `Resize.scale`、
以及 `test_pipeline` 中的 `Resize.scale`。解码器输出的最后一层用的是
`UpsamplingBilinear2d(scale_factor=8)`，分辨率变化时建议检查输出尺寸是否与掩码对齐。
</details>

<details>
<summary><b>训练显存不够？</b></summary>

降低 `train_dataloader.batch_size`，或加上 `--amp` 开启混合精度。
</details>

<details>
<summary><b>环境里装过原版 MMSegmentation，装 DADNet 后 <code>import mmseg</code> 拿到的是旧的那个</b></summary>

DADNet 和 MMSegmentation 提供的是**同一个 Python 包名 `mmseg`**，两者不能共存。
先卸掉原版再装本仓库：

```bash
pip uninstall mmsegmentation -y
pip install -v -e .
```

如果原版是以 editable 方式安装的、指向另一个目录，也要一并清理：

```bash
pip freeze | grep -i mmseg       # 找到残留条目
python -c "import mmseg; print(mmseg.__file__)"   # 确认最终指向本仓库
```

临时验证（不想动环境）可以用 `PYTHONPATH` 覆盖：

```bash
PYTHONPATH=/path/to/DADNET python tools/train.py ...
```
</details>

---

## 8. 引用

如果本工作对你的研究有帮助，请引用：

```bibtex
@article{dadnet,
  title   = {DADNet: Dual-Attention U-Net for Industrial Surface Defect Segmentation},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

本仓库基于 MMSegmentation，也请一并引用：

```bibtex
@misc{mmseg2020,
  title     = {{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
  author    = {MMSegmentation Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
  year      = {2020}
}
```

---

## 9. 致谢与许可

本仓库是 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.2.2 的
衍生作品，遵循 **Apache License 2.0**，详见 [LICENSE](LICENSE)。
对上游代码的修改记录见 [NOTICE](NOTICE)。

数据集版权归原作者所有，本仓库不分发数据，仅提供准备脚本。使用前请自行获取并
遵守其许可条款。

感谢 OpenMMLab 社区。
