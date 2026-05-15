import argparse
import csv
import json
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
    parser.add_argument('--image_path', default='',
                        help='input RGB image for SPAM attention visualization')
    parser.add_argument('--sentence', default='',
                        help='referring expression used for mask generation')
    parser.add_argument('--attention_output_dir', required=True,
                        help='directory where heatmaps and token scores are written')
    parser.add_argument('--batch_test_all', action='store_true',
                        help='visualize every sample in the PlantSeg split specified by --batch_split')
    parser.add_argument('--batch_split', default='test',
                        help='PlantSeg split to use when --batch_test_all is set')
    parser.add_argument('--batch_limit', default=0, type=int,
                        help='optional maximum number of samples for batch visualization; 0 means all')
    parser.add_argument('--overlay_alpha', default=0.80, type=float,
                        help='heatmap overlay opacity in [0, 1]')
    return parser


def normalize_map(array):
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value + 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def colorize_heatmap(heatmap):
    heatmap = normalize_map(heatmap)
    low = np.array([0, 0, 255], dtype=np.float32)
    high = np.array([255, 0, 0], dtype=np.float32)
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
    h, w = infer_square_hw(hw)
    token_weighted = (sim_map.mean(dim=1) * token_importance.squeeze(-1).unsqueeze(1)).sum(dim=-1)

    return {
        'height': h,
        'width': w,
        'token_importance': token_importance.squeeze(-1).detach().cpu(),
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


def prepare_inputs(args, tokenizer, device, image_path, sentence):
    original = Image.open(image_path).convert('RGB')
    transform = TV.Compose([
        TV.Resize((args.img_size, args.img_size)),
        TV.ToTensor(),
        TV.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    image = transform(original).unsqueeze(0).to(device)
    input_ids, attention_mask = tokenize_text(tokenizer, sentence, args.max_text_tokens)
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


def export_spam_maps(records, original_image, output_dir, alpha):
    summary = {}
    aggregate_weighted = []

    for stage_name, record in records.items():
        stage_key = stage_name.replace('.', '_')
        h, w = record['height'], record['width']
        weighted = record['token_weighted_attention'][0].reshape(h, w).numpy()
        output_path = output_dir / 'spam_{}_token_weighted.png'.format(stage_key)
        save_heatmap_overlay(original_image, weighted, output_path, alpha)
        aggregate_weighted.append(resize_array(weighted, original_image.size))

        summary[stage_name] = {
            'spatial_size': [h, w],
            'spam_token_weighted': str(output_path),
        }

    if aggregate_weighted:
        all_stage_weighted = np.stack(aggregate_weighted, axis=0).mean(axis=0)
        output_path = output_dir / 'spam_all_stages_token_weighted.png'
        save_heatmap_overlay(original_image, all_stage_weighted, output_path, alpha)
        summary['all_stages'] = {
            'spam_token_weighted': str(output_path),
        }

    return summary


def export_hlg_maps(aux_outputs, original_image, output_dir, alpha):
    summary = {}
    if not aux_outputs:
        return summary

    aggregate = []
    for name, tensor in sorted(aux_outputs.items()):
        if not name.startswith('hlg_stage'):
            continue
        heatmap = tensor.detach().cpu()[0].mean(dim=0).numpy()
        output_path = output_dir / '{}_suppression.png'.format(name)
        save_heatmap_overlay(original_image, heatmap, output_path, alpha)
        aggregate.append(resize_array(heatmap, original_image.size))
        summary[name] = str(output_path)

    if aggregate:
        output_path = output_dir / 'hlg_all_stages_suppression.png'
        save_heatmap_overlay(original_image, np.stack(aggregate, axis=0).mean(axis=0), output_path, alpha)
        summary['all_hlg_stages'] = str(output_path)
    return summary


def write_raw_arrays(records, aux_outputs, tokens, valid_mask, output_dir):
    arrays = {
        'tokens': np.array(tokens),
        'valid_token_mask': np.array(valid_mask, dtype=np.bool_),
    }
    for stage_name, record in records.items():
        stage_key = stage_name.replace('.', '_')
        h, w = record['height'], record['width']
        arrays['spam_spatial_size__{}'.format(stage_key)] = np.array([h, w], dtype=np.int32)
        arrays['token_importance__{}'.format(stage_key)] = record['token_importance'][0].numpy()
        arrays['spam_token_weighted__{}'.format(stage_key)] = \
            record['token_weighted_attention'][0].reshape(h, w).numpy()
    for name, tensor in sorted(aux_outputs.items()):
        if name.startswith('hlg_stage'):
            arrays['hlg_suppression__{}'.format(name)] = tensor.detach().cpu()[0].mean(dim=0).numpy()

    output_path = output_dir / 'attention_raw_arrays.npz'
    np.savez_compressed(output_path, **arrays)
    return output_path


def run_one_sample(args, model, text_encoder, tokenizer, device, sample_id, image_path, sentence, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    original, image, input_ids, attention_mask, tokens = prepare_inputs(args, tokenizer, device, image_path, sentence)
    records = {}
    handles = register_spam_hooks(model, records)
    if not handles:
        raise RuntimeError('No SPAM modules were found. Use --ablation_config spam_only or --ablation_config ours.')

    try:
        with torch.no_grad():
            _output, aux_outputs = forward_model(model, text_encoder, image, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    if not records:
        raise RuntimeError('SPAM hooks did not record attention. Check that the loaded model actually uses SPAM.')

    valid_mask = attention_mask.squeeze(0).detach().cpu().bool().tolist()
    token_csv = write_token_scores(records, tokens, valid_mask, output_dir)
    spam_summary = export_spam_maps(records, original, output_dir, args.overlay_alpha)
    hlg_summary = export_hlg_maps(aux_outputs, original, output_dir, args.overlay_alpha)
    raw_npz = write_raw_arrays(records, aux_outputs, tokens, valid_mask, output_dir)
    metadata = {
        'sample_id': sample_id,
        'image_path': str(Path(image_path).expanduser().resolve()),
        'sentence': sentence,
        'resume': args.resume,
        'ablation_config': args.ablation_config,
        'align_module': args.align_module,
        'gate_module': args.gate_module,
        'hlg_stages': args.hlg_stages,
        'tokens': tokens,
        'valid_token_mask': valid_mask,
        'token_scores_csv': str(token_csv),
        'raw_arrays_npz': str(raw_npz),
        'spam_attention': spam_summary,
        'hlg_suppression': hlg_summary,
    }
    with (output_dir / 'spam_attention_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


def load_batch_samples(args):
    if args.dataset != 'plantseg':
        raise ValueError('--batch_test_all currently requires --dataset plantseg.')
    root = Path(args.plantseg_root).expanduser().resolve()
    metadata_path = root / 'main.json'
    with metadata_path.open('r', encoding='utf-8') as handle:
        records = json.load(handle)
    samples = [record for record in records if record.get('split') == args.batch_split]
    if args.batch_limit > 0:
        samples = samples[:args.batch_limit]
    if not samples:
        raise ValueError('No samples found for split [{}] under {}'.format(args.batch_split, metadata_path))
    return samples


def main(args):
    if not args.resume:
        raise ValueError('--resume is required for trained-weight SPAM attention visualization.')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_root = Path(args.attention_output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model, text_encoder, tokenizer = build_model_and_text(args, device)

    if args.batch_test_all:
        samples = load_batch_samples(args)
        batch_csv = output_root / 'batch_attention_summary.csv'
        with batch_csv.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['index', 'sample_id', 'image_path', 'output_dir', 'status', 'message'])
            for index, sample in enumerate(samples):
                sample_id = str(sample.get('id') or 'sample_{:06d}'.format(index))
                image_path = Path(args.plantseg_root).expanduser().resolve() / sample['image']
                captions = sample.get('caption') or []
                if len(captions) <= args.plantseg_caption_index:
                    raise IndexError('Sample [{}] does not have caption index {}'.format(
                        sample_id, args.plantseg_caption_index))
                sentence = captions[args.plantseg_caption_index]
                sample_output_dir = output_root / sample_id
                try:
                    run_one_sample(args, model, text_encoder, tokenizer, device,
                                   sample_id, image_path, sentence, sample_output_dir)
                    writer.writerow([index, sample_id, str(image_path), str(sample_output_dir), 'ok', ''])
                    print('[{}/{}] wrote {}'.format(index + 1, len(samples), sample_output_dir))
                except Exception as exc:
                    writer.writerow([index, sample_id, str(image_path), str(sample_output_dir), 'error', str(exc)])
                    print('[{}/{}] failed {}: {}'.format(index + 1, len(samples), sample_id, exc))
        print('Wrote batch SPAM attention summary to {}'.format(batch_csv))
        return

    if not args.image_path or not args.sentence:
        raise ValueError('Single-sample mode requires --image_path and --sentence. '
                         'Use --batch_test_all to process the PlantSeg test split automatically.')
    run_one_sample(args, model, text_encoder, tokenizer, device,
                   Path(args.image_path).stem, args.image_path, args.sentence, output_root)
    print('Wrote SPAM attention visualization to {}'.format(output_root))


if __name__ == '__main__':
    parser = add_visualization_args(get_parser())
    args = validate_args(parser.parse_args())
    main(args)
