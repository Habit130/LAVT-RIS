# Linux 4090 运行说明

本仓库的服务器交付面锁定为：

- Linux
- 单张 RTX 4090
- CUDA 11.8
- Python 3.10
- Miniconda
- 数据目录为仓库同级的 `../plantseg`
- 文本输入固定使用 `caption[3]`
- 训练主模型固定为 `lavt_one`

## 1. 环境

使用 `environment.server.yml` 创建 Conda 环境。

## 2. 外部资产

- `bert-base-uncased`：沿用 Hugging Face 名称下载。
- Swin 初始化权重：使用与原仓库 `swin_base_patch4_window12_384_22k.pth` 张量布局兼容的 Hugging Face 直链或本地文件。
- `plantseg`：保持为仓库同级目录，结构固定为：
  - `../plantseg/main.json`
  - `../plantseg/images/`
  - `../plantseg/ann/`

## 3. 训练

单卡训练命令面：

```bash
python train.py \
  --model lavt_one \
  --dataset plantseg \
  --model_id plantseg_lavt_one \
  --batch-size 4 \
  --lr 0.00005 \
  --wd 1e-2 \
  --swin_type base \
  --window12 \
  --img_size 480 \
  --epochs 40 \
  --plantseg_root ../plantseg \
  --plantseg_caption_index 3 \
  --pretrained_swin_weights <hf-or-local-swin-weight>
```

如果 4090 实测 OOM，唯一允许的回退是把 `--batch-size` 改成 `2`。

训练输出：

- 最佳 checkpoint：`./checkpoints/model_best_<model_id>.pth`
- 选模指标：`val` split 上的前景 `IoU`

## 4. 训练后验证

正式测试命令面：

```bash
python test.py \
  --model lavt_one \
  --dataset plantseg \
  --split test \
  --swin_type base \
  --window12 \
  --img_size 480 \
  --workers 4 \
  --plantseg_root ../plantseg \
  --plantseg_caption_index 3 \
  --resume ./checkpoints/model_best_plantseg_lavt_one.pth
```

输出指标固定为：

- `IoU`
- `Dice`
- `Recall`
- `mIoU`
- `mACC`

