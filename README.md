# DADNet

**DADNet: A Dual-branch Attention Differential Network for Automated Industrial Surface Defect Detection**

基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.2.2 实现，在 NEU-Seg 热轧带钢缺陷数据集上验证。

## 目录

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

## 2. 数据集准备

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
`NEU-seg` 从这些 JSON 读取图像-掩码配对，而不是扫描目录。


## 3. 训练


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


## 4. 测试与评估

```bash
python tools/test.py configs/dadnet/dadnet_unet32_neu_seg.py \
    work_dirs/dadnet_unet32_neu_seg/best_mIoU_iter_XXXX.pth
```

## 5. 引用

如果本工作对你的研究有帮助，请引用：

```bibtex
@article{he2026dadnet,
  title={DADNet: A Dual-branch Attention Differential Network for Automated Industrial Surface Defect Detection},
  author={He, Yingmei and Tian, Bingbing and Ma, Yunfeng and Zhang, Yiqiong and Wang, Xueping and Wang, Yaonan and Liu, Min},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
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
