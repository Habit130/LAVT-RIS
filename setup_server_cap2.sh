#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
ENV_NAME="${ENV_NAME:-lavt-plantseg}"
CONDA_SH="${CONDA_SH:-/usr/local/miniconda3/etc/profile.d/conda.sh}"
PLANTSEG_ROOT="${PLANTSEG_ROOT:-$REPO_ROOT/../plantseg}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO_ROOT/pretrained_weights}"
BERT_DIR="${BERT_DIR:-$WEIGHTS_DIR/bert-base-uncased}"
SWIN_PATH="${SWIN_PATH:-$WEIGHTS_DIR/swin_base_patch4_window12_384_22k.pth}"
MODEL_ID="${MODEL_ID:-plantseg_lavt_one_cap2}"
MASK_SAVE_DIR="${MASK_SAVE_DIR:-$REPO_ROOT/pred_masks_cap2}"

SWIN_URL="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth"
BERT_MODEL_URL="https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"
BERT_CONFIG_URL="https://huggingface.co/bert-base-uncased/resolve/main/config.json"
BERT_VOCAB_URL="https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local target="$2"
  if [[ -s "$target" ]]; then
    echo "[skip] $target"
    return 0
  fi

  mkdir -p "$(dirname "$target")"
  local tmp="${target}.tmp"
  rm -f "$tmp"
  echo "[download] $url -> $target"
  curl -L --retry 5 --retry-delay 5 --fail "$url" -o "$tmp"
  mv "$tmp" "$target"
}

resolve_conda_sh() {
  if [[ -f "$CONDA_SH" ]]; then
    return 0
  fi

  if [[ -n "${CONDA_EXE:-}" ]]; then
    local guessed
    guessed="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)/etc/profile.d/conda.sh"
    if [[ -f "$guessed" ]]; then
      CONDA_SH="$guessed"
      return 0
    fi
  fi

  echo "Cannot find conda.sh. Set CONDA_SH before running this script." >&2
  exit 1
}

main() {
  require_cmd curl
  require_cmd bash

  cd "$REPO_ROOT"

  if [[ ! -f "$REPO_ROOT/environment.server.yml" ]]; then
    echo "Missing $REPO_ROOT/environment.server.yml" >&2
    exit 1
  fi

  resolve_conda_sh
  # shellcheck disable=SC1090
  source "$CONDA_SH"

  if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "[conda] updating existing environment: $ENV_NAME"
    conda env update -n "$ENV_NAME" -f "$REPO_ROOT/environment.server.yml" --prune
  else
    echo "[conda] creating environment from environment.server.yml"
    conda env create -f "$REPO_ROOT/environment.server.yml"
  fi

  conda activate "$ENV_NAME"
  conda install -y numpy=1.26.4

  mkdir -p "$WEIGHTS_DIR" "$BERT_DIR"
  download_file "$SWIN_URL" "$SWIN_PATH"
  download_file "$BERT_MODEL_URL" "$BERT_DIR/pytorch_model.bin"
  download_file "$BERT_CONFIG_URL" "$BERT_DIR/config.json"
  download_file "$BERT_VOCAB_URL" "$BERT_DIR/vocab.txt"

  python - <<'PY'
import numpy
import torch
print("numpy=", numpy.__version__)
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
PY

  if [[ -f "$PLANTSEG_ROOT/main.json" ]]; then
    echo "[dataset] found $PLANTSEG_ROOT/main.json"
  else
    echo "[dataset] warning: $PLANTSEG_ROOT/main.json not found yet" >&2
  fi

  cat <<EOF

Setup complete.

Repo root:
  $REPO_ROOT

Conda env:
  $ENV_NAME

Weights:
  Swin: $SWIN_PATH
  BERT: $BERT_DIR

PlantSeg root:
  $PLANTSEG_ROOT

Caption index for this setup:
  2

Train command:
  cd "$REPO_ROOT" && source "$CONDA_SH" && conda activate "$ENV_NAME" && python train.py --model lavt_one --dataset plantseg --model_id "$MODEL_ID" --batch-size 4 --lr 1e-5 --wd 1e-2 --swin_type base --window12 --img_size 480 --epochs 40 --workers 4 --pin_mem --device cuda:0 --plantseg_root "$PLANTSEG_ROOT" --plantseg_caption_index 2 --ck_bert "$BERT_DIR" --bert_tokenizer "$BERT_DIR" --pretrained_swin_weights "$SWIN_PATH"

Test command:
  cd "$REPO_ROOT" && source "$CONDA_SH" && conda activate "$ENV_NAME" && python test.py --model lavt_one --dataset plantseg --split test --swin_type base --window12 --img_size 480 --workers 4 --device cuda:0 --plantseg_root "$PLANTSEG_ROOT" --plantseg_caption_index 2 --ck_bert "$BERT_DIR" --bert_tokenizer "$BERT_DIR" --resume "$REPO_ROOT/checkpoints/model_best_${MODEL_ID}.pth" --save_mask_dir "$MASK_SAVE_DIR"

EOF
}

main "$@"
