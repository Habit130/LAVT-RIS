import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TV
from PIL import Image

from args import get_parser, validate_args
from lib import segmentation
from lib.backbone import SPAM
from test import validate_checkpoint_config
from text_encoder import (build_text_encoder, build_text_tokenizer, encode_text,
                          get_checkpoint_text_encoder_state, prepare_text_encoder_args,
                          tokenize_text)
import utils


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def add_visualization_args(parser):
    parser.add_argument('--image_path', required=True,
                        help='input RGB image for SPAM attention visualization')
    parser.add_argument('--sentence', required=True,
                        help='referring expression used for mask generation')
    parser.add_argument('--attention_output_dir', required=True,
                        help='directory where heatmaps and token scores are written')
    parser.add_argument('--top_tokens', default=8, type=int,
                        help='number of highest-importance valid tokens to render per SPAM stage')
    parser.add_argument('--overlay_alpha', default=0.55, type=float,
                        help='heatmap overlay opacity in [0, 1]')
    return parser


def safe_name(text):
    text = text.replace('##', '')
    text = re.sub(r'[^0-9A-Za-z._-]+', '_', text)
    return text.strip('_') or 'token'


def normalize_map(array):
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value + 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def colorize_heatmap(heatmap):
    heatmap = normalize_map(heatmap)
    low = np.array([37, 99, 235], dtype=np.float32)
    high = np.array([239, 68, 68], dtype=np.float32)
    rgb = low[None, None, :] * (1.0 - heatmap[..., None]) + high[None, None, :] * heatmap[..., None]
    return rgb.clip(0, 255).astype(np.uint8)


def resize_array(array, size):
    image = Image.fromarray((normalize_map(array) * 255).astype(np.uint8), mode='L')
    image = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(image).astype(np.float32) / 255.0


def save_heatmap_overlay(original_image, heatmap, output_path, alpha):
    heatmap_resized = resize_array(heatmap, original_image.size)
    color = colorize_heatmap(heatmap_resized)
    base = np.asarray(original_image).astype(np.float32)
    overlay = ((1.0 - alpha) * base + alpha * color).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


def save_prediction_overlay(original_image, prediction, output_path, alpha=0.45):
    pred = Image.fromarray((prediction.astype(np.uint8) * 255), mode='L')
    pred = pred.resize(original_image.size, resample=Image.NEAREST)
    pred_array = np.asarray(pred) > 0
    base = np.asarray(original_image).astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    overlay = base.copy()
    overlay[pred_array] = ((1.0 - alpha) * base[pred_array] + alpha * red[pred_array])
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(output_path)


def infer_square_hw(hw):
    side = int(round(hw ** 0.5))
    if side * side != hw:
        raise ValueError('Cannot infer square spatial size from HW={}'.format(hw))
    return side, side


def recompute_spam_attention(module, inputs):
    x, l, l_mask = inputs
    if l_mask.dim() == 2:
        token_mask = l_mask.unsqueeze(-1)
    elif l_mask.dim() == 3:
        token_mask = l_mask
    else:
        raise RuntimeError('Unexpected SPAM l_mask shape {}'.format(tuple(l_mask.shape)))

    bsz, hw = x.size(0), x.size(1)
    l_tokens = l.permute(0, 2, 1).contiguous()
    token_mask = token_mask.to(device=l_tokens.device, dtype=l_tokens.dtype)

    visual_summary = x.mean(dim=1)
    token_context = module.token_text_proj(l_tokens) + module.token_visual_proj(visual_summary).unsqueeze(1)
    token_importance = module.token_score(token_context) * token_mask

    x_channels = x.permute(0, 2, 1).contiguous()
    query = module.f_query(x_channels).permute(0, 2, 1).contiguous()
    mask_channels = token_mask.permute(0, 2, 1).contiguous()
    key = module.f_key(l) * mask_channels

    n_l = key.size(-1)
    head_key_channels = module.key_channels // module.num_heads
    query = query.reshape(bsz, hw, module.num_heads, head_key_channels).permute(0, 2, 1, 3)
    key = key.reshape(bsz, module.num_heads, head_key_channels, n_l)

    sim_map = torch.matmul(query, key) * (head_key_channels ** -0.5)
    sim_map = sim_map.masked_fill(mask_channels.unsqueeze(1) <= 0, -1e4)
    sim_map = F.softmax(sim_map, dim=-1)
    anomaly_response = module.anomaly_score(x)

    h, w = infer_square_hw(hw)
    token_weighted = (sim_map.mean(dim=1) * token_importance.squeeze(-1).unsqueeze(1)).sum(dim=-1)

    return {
        'height': h,
        'width': w,
        'sim_map': sim_map.detach().cpu(),
        'token_importance': token_importance.squeeze(-1).detach().cpu(),
        'anomaly_response': anomaly_response.squeeze(-1).detach().cpu(),
        'token_weighted_attention': token_weighted.detach().cpu(),
    }


def register_spam_hooks(model, records):
    handles = []

    def make_hook(name):
        def hook(module, inputs, _output):
            with torch.no_grad():
                records[name] = recompute_spam_attention(module, inputs)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, SPAM):
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def build_model_and_text(args, device):
    text_encoder_config = prepare_text_encoder_args(args)
    model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    validate_checkpoint_config(checkpoint, args, args.resume, context='SPAM attention')
    utils.load_state_dict_with_fallback(model, checkpoint['model'], strict=True, description='model')
    model = model.to(device).eval()

    if args.model == 'lavt_one':
        text_encoder = None
    else:
        text_encoder = build_text_encoder(args, config=text_encoder_config)
        text_encoder_state, text_encoder_key = get_checkpoint_text_encoder_state(checkpoint, args)
        if text_encoder_state is None:
            raise KeyError('Checkpoint [{}] is missing text encoder weights.'.format(args.resume))
        utils.load_state_dict_with_fallback(text_encoder, text_encoder_state, strict=True,
                                            description=text_encoder_key)
        text_encoder = text_encoder.to(device).eval()

    tokenizer = build_text_tokenizer(args)
    return model, text_encoder, tokenizer


def prepare_inputs(args, tokenizer, device):
    original = Image.open(args.image_path).convert('RGB')
    transform = TV.Compose([
        TV.Resize((args.img_size, args.img_size)),
        TV.ToTensor(),
        TV.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    image = transform(original).unsqueeze(0).to(device)
    input_ids, attention_mask = tokenize_text(tokenizer, args.sentence, args.max_text_tokens)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())
    return original, image, input_ids, attention_mask, tokens


def forward_model(model, text_encoder, image, input_ids, attention_mask):
    if text_encoder is not None:
        encoded = encode_text(text_encoder, input_ids, attention_mask)
        embedding = encoded.permute(0, 2, 1)
        return model(image, embedding, l_mask=attention_mask.unsqueeze(-1), return_aux=True)
    return model(image, input_ids, l_mask=attention_mask, return_aux=True)


def write_token_scores(records, tokens, valid_mask, output_dir):
    csv_path = output_dir / 'spam_token_scores.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['stage', 'token_index', 'token', 'valid', 'token_importance'])
        for stage_name, record in records.items():
            scores = record['token_importance'][0].numpy()
            for index, score in enumerate(scores):
                writer.writerow([stage_name, index, tokens[index], int(valid_mask[index]), float(score)])
    return csv_path


def export_stage_maps(records, tokens, valid_mask, original_image, output_dir, top_k, alpha):
    summary = {}
    valid_indices = [idx for idx, valid in enumerate(valid_mask) if valid]
    aggregate_weighted = []
    aggregate_anomaly = []

    for stage_name, record in records.items():
        stage_dir = output_dir / stage_name.replace('.', '_')
        stage_dir.mkdir(parents=True, exist_ok=True)
        h, w = record['height'], record['width']
        token_scores = record['token_importance'][0].numpy()

        anomaly = record['anomaly_response'][0].reshape(h, w).numpy()
        weighted = record['token_weighted_attention'][0].reshape(h, w).numpy()
        save_heatmap_overlay(original_image, anomaly, stage_dir / 'image_focus_anomaly_response.png', alpha)
        save_heatmap_overlay(original_image, weighted, stage_dir / 'image_focus_token_weighted.png', alpha)
        aggregate_anomaly.append(resize_array(anomaly, original_image.size))
        aggregate_weighted.append(resize_array(weighted, original_image.size))

        ranked = sorted(valid_indices, key=lambda idx: float(token_scores[idx]), reverse=True)
        ranked = ranked[:max(0, top_k)]
        stage_rows = []
        sim_mean = record['sim_map'][0].mean(dim=0)
        for rank, token_index in enumerate(ranked, start=1):
            token_map = sim_mean[:, token_index].reshape(h, w).numpy()
            token_name = safe_name(tokens[token_index])
            out_name = 'token_{:02d}_idx{:02d}_{}.png'.format(rank, token_index, token_name)
            save_heatmap_overlay(original_image, token_map, stage_dir / out_name, alpha)
            stage_rows.append({
                'rank': rank,
                'token_index': token_index,
                'token': tokens[token_index],
                'token_importance': float(token_scores[token_index]),
                'heatmap': str(stage_dir / out_name),
            })

        summary[stage_name] = {
            'spatial_size': [h, w],
            'image_focus_anomaly_response': str(stage_dir / 'image_focus_anomaly_response.png'),
            'image_focus_token_weighted': str(stage_dir / 'image_focus_token_weighted.png'),
            'top_tokens': stage_rows,
        }

    if aggregate_weighted:
        all_stage_weighted = np.stack(aggregate_weighted, axis=0).mean(axis=0)
        save_heatmap_overlay(original_image, all_stage_weighted,
                             output_dir / 'image_focus_all_spam_stages_token_weighted.png', alpha)
    if aggregate_anomaly:
        all_stage_anomaly = np.stack(aggregate_anomaly, axis=0).mean(axis=0)
        save_heatmap_overlay(original_image, all_stage_anomaly,
                             output_dir / 'image_focus_all_spam_stages_anomaly_response.png', alpha)

    return summary


def export_hlg_maps(aux_outputs, original_image, output_dir, alpha):
    summary = {}
    if not aux_outputs:
        return summary

    hlg_dir = output_dir / 'hlg_suppression'
    hlg_dir.mkdir(parents=True, exist_ok=True)
    aggregate = []
    for name, tensor in sorted(aux_outputs.items()):
        if not name.startswith('hlg_stage'):
            continue
        heatmap = tensor.detach().cpu()[0].mean(dim=0).numpy()
        output_path = hlg_dir / '{}_suppression.png'.format(name)
        save_heatmap_overlay(original_image, heatmap, output_path, alpha)
        aggregate.append(resize_array(heatmap, original_image.size))
        summary[name] = str(output_path)

    if aggregate:
        output_path = hlg_dir / 'all_hlg_stages_suppression.png'
        save_heatmap_overlay(original_image, np.stack(aggregate, axis=0).mean(axis=0), output_path, alpha)
        summary['all_hlg_stages'] = str(output_path)
    return summary


def main(args):
    if not args.resume:
        raise ValueError('--resume is required for trained-weight SPAM attention visualization.')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.attention_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, text_encoder, tokenizer = build_model_and_text(args, device)
    original, image, input_ids, attention_mask, tokens = prepare_inputs(args, tokenizer, device)

    records = {}
    handles = register_spam_hooks(model, records)
    if not handles:
        raise RuntimeError('No SPAM modules were found. Use --ablation_config spam_only or --ablation_config ours.')

    with torch.no_grad():
        output, aux_outputs = forward_model(model, text_encoder, image, input_ids, attention_mask)
        prediction = output.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    for handle in handles:
        handle.remove()

    if not records:
        raise RuntimeError('SPAM hooks did not record attention. Check that the loaded model actually uses SPAM.')

    save_prediction_overlay(original, prediction, output_dir / 'prediction_overlay.png')
    valid_mask = attention_mask.squeeze(0).detach().cpu().bool().tolist()
    token_csv = write_token_scores(records, tokens, valid_mask, output_dir)
    summary = export_stage_maps(records, tokens, valid_mask, original, output_dir,
                                args.top_tokens, args.overlay_alpha)
    hlg_summary = export_hlg_maps(aux_outputs, original, output_dir, args.overlay_alpha)
    metadata = {
        'image_path': str(Path(args.image_path).expanduser().resolve()),
        'sentence': args.sentence,
        'resume': args.resume,
        'ablation_config': args.ablation_config,
        'align_module': args.align_module,
        'gate_module': args.gate_module,
        'hlg_stages': args.hlg_stages,
        'tokens': tokens,
        'valid_token_mask': valid_mask,
        'prediction_overlay': str(output_dir / 'prediction_overlay.png'),
        'image_focus_all_spam_stages_token_weighted': str(
            output_dir / 'image_focus_all_spam_stages_token_weighted.png'),
        'image_focus_all_spam_stages_anomaly_response': str(
            output_dir / 'image_focus_all_spam_stages_anomaly_response.png'),
        'token_scores_csv': str(token_csv),
        'stages': summary,
        'hlg_suppression': hlg_summary,
    }
    with (output_dir / 'spam_attention_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print('Wrote SPAM attention visualization to {}'.format(output_dir))


if __name__ == '__main__':
    parser = add_visualization_args(get_parser())
    args = validate_args(parser.parse_args())
    main(args)
