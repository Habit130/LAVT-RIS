# Linux 4090 运行说明

本仓库当前锁定的服务器交付面为：

- Linux
- 单张 RTX 4090
- CUDA 11.8
- Python 3.10
- Miniconda
- 数据目录为仓库同级的 `../plantseg`
- 文本输入默认使用 `caption[3]`
- 训练主模型固定为 `lavt_one`
- 默认文本编码器为 `microsoft/deberta-v3-base`
- 默认文本长度为 `64`

## 1. 环境

使用 `environment.server.yml` 创建 Conda 环境。

## 2. 外部资产

- DeBERTa-v3-base：训练和测试时通过 Hugging Face 名称 `microsoft/deberta-v3-base` 加载
- Swin 初始化权重：使用与原仓库兼容的 `swin_base_patch4_window12_384_22k.pth`
- `plantseg`：保持为仓库同级目录，结构固定为：
  - `../plantseg/main.json`
  - `../plantseg/images/`
  - `../plantseg/ann/`

## 3. 训练

单卡训练命令为：

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
  --text_encoder_name microsoft/deberta-v3-base \
  --max_text_tokens 64 \
  --pretrained_swin_weights <hf-or-local-swin-weight>
```

如果 4090 实测 OOM，唯一允许的回退是把 `--batch-size` 改成 `2`。

训练输出：

- 最佳 checkpoint：`./checkpoints/model_best_<model_id>.pth`
- 选模指标：`val` split 上的前景 `IoU`

## 4. 训练后验证

正式测试命令为：

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
  --text_encoder_name microsoft/deberta-v3-base \
  --max_text_tokens 64 \
  --resume ./checkpoints/model_best_plantseg_lavt_one.pth
```

输出指标固定为：

- `IoU`
- `Dice`
- `Recall`
- `mIoU`
- `mACC`

## 5. 严格对照设置

如果需要做“只替换文本编码器，不改变旧长度策略”的严格对照，请显式指定：

```bash
--max_text_tokens 20
```
