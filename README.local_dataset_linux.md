# LAVT-RIS Local JSON Dataset + Linux 4090 Workflow

## What changed
- Training and testing now read `../dataset/train.json` and `../dataset/test.json` by default.
- The original `refer` dataset API is no longer required for the main training/testing path.
- Training no longer runs a validation split each epoch.
- Runtime dependencies on `mmcv-full` and `mmsegmentation` were removed.
- The bundled BERT tokenizer now supports slow-tokenizer mode even if the `tokenizers` wheel is absent.

## Expected dataset layout
The repository root is assumed to be `LAVT-RIS/`, and the dataset is assumed to live beside it:

```text
Segmentation/
  LAVT-RIS/
  dataset/
    train.json
    test.json
    train/
      img/
      lbl/
    test/
      img/
      lbl/
```

Each JSON item must contain at least these fields:

```json
{
  "id": "sample_id",
  "image": "train/img/example.jpg",
  "mask": "train/lbl/example.png",
  "caption": ["the abnormal region", "... optional extra expressions ..."]
}
```

`image` and `mask` may be absolute paths or paths relative to `--dataset_root`.

## Mask handling
Masks are converted to binary segmentation targets inside the dataset loader.
All non-zero pixels are treated as foreground, so original class ids such as `1`, `2`, ..., `110+` are all collapsed to `1`.

## Linux environment
Create the conda environment:

```bash
conda env create -f environment.linux.4090.yml
conda activate lavt-ris-linux
```

## External assets
This repository still expects external weights and does not track them in Git:
- Swin backbone initialization weights: pass the local path with `--pretrained_swin_weights`
- BERT weights/tokenizer: keep using `bert-base-uncased` with internet access, or pass a local HuggingFace-style directory to both `--ck_bert` and `--bert_tokenizer`

## Train on Linux
Use `torchrun` even on a single GPU:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --model lavt \
  --model_id local_json_run \
  --dataset_root ../dataset \
  --train_json train.json \
  --test_json test.json \
  --batch-size 4 \
  --lr 5e-5 \
  --wd 1e-2 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --epochs 40 \
  --img_size 480 \
  --pin_mem
```

Training saves:
- `./checkpoints/checkpoint_last_<model_id>.pth`: latest resumable checkpoint
- `./checkpoints/model_final_<model_id>.pth`: final checkpoint after training

Resume training:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --model lavt \
  --model_id local_json_run \
  --dataset_root ../dataset \
  --resume ./checkpoints/checkpoint_last_local_json_run.pth \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth
```

## Test on the held-out split
Testing uses `test.json` by default:

```bash
python test.py \
  --model lavt \
  --swin_type base \
  --dataset_root ../dataset \
  --test_json test.json \
  --split test \
  --resume ./checkpoints/model_final_local_json_run.pth \
  --workers 4 \
  --ddp_trained_weights \
  --window12 \
  --img_size 480
```

## Lightweight dataset sanity check
Run this before training if you want to verify the JSON files and file paths:

```bash
python tools/check_dataset_json.py --dataset_root ../dataset
```
