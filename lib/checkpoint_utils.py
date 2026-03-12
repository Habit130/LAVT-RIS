import logging
import os.path as osp

import torch
from torch.nn import functional as F


def get_root_logger():
    logger = logging.getLogger('lavt')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError('No state_dict found in checkpoint file')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    if not state_dict:
        raise RuntimeError('Checkpoint contains an empty state_dict')

    first_key = next(iter(state_dict))
    if first_key.startswith('module.'):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
        first_key = next(iter(state_dict))
    if first_key.startswith('backbone.'):
        state_dict = {
            key.replace('backbone.', ''): value
            for key, value in state_dict.items()
            if key.startswith('backbone.')
        }
        first_key = next(iter(state_dict))
    if first_key.startswith('encoder.'):
        state_dict = {
            key.replace('encoder.', ''): value
            for key, value in state_dict.items()
            if key.startswith('encoder.')
        }

    return state_dict


def _resize_position_embeddings(model, state_dict, logger):
    if state_dict.get('absolute_pos_embed') is not None and hasattr(model, 'absolute_pos_embed'):
        absolute_pos_embed = state_dict['absolute_pos_embed']
        n1, length, channels = absolute_pos_embed.size()
        n2, c2, height, width = model.absolute_pos_embed.size()
        if n1 == n2 and channels == c2 and length == height * width:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(n2, height, width, c2).permute(0, 3, 1, 2)
        else:
            logger.warning('Skipping absolute_pos_embed because checkpoint and model shapes do not match.')
            state_dict.pop('absolute_pos_embed')

    relative_keys = [key for key in state_dict.keys() if 'relative_position_bias_table' in key]
    model_state = model.state_dict()
    for key in relative_keys:
        if key not in model_state:
            continue
        pretrained_table = state_dict[key]
        current_table = model_state[key]
        length_pretrained, num_heads_pretrained = pretrained_table.size()
        length_current, num_heads_current = current_table.size()
        if num_heads_pretrained != num_heads_current:
            logger.warning('Skipping %s because the attention head count does not match.', key)
            state_dict.pop(key)
            continue
        if length_pretrained == length_current:
            continue

        size_pretrained = int(length_pretrained ** 0.5)
        size_current = int(length_current ** 0.5)
        resized = F.interpolate(
            pretrained_table.permute(1, 0).view(1, num_heads_pretrained, size_pretrained, size_pretrained),
            size=(size_current, size_current),
            mode='bicubic',
        )
        state_dict[key] = resized.view(num_heads_current, length_current).permute(1, 0)


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    if logger is None:
        logger = get_root_logger()

    if not osp.isfile(filename):
        raise IOError(f'{filename} is not a checkpoint file')

    checkpoint = torch.load(filename, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint)
    _resize_position_embeddings(model, state_dict, logger)

    load_result = model.load_state_dict(state_dict, strict=False)
    if isinstance(load_result, tuple):
        missing_keys, unexpected_keys = load_result
    else:
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys

    if strict and (missing_keys or unexpected_keys):
        error_parts = []
        if unexpected_keys:
            error_parts.append('unexpected keys: {}'.format(', '.join(unexpected_keys)))
        if missing_keys:
            error_parts.append('missing keys: {}'.format(', '.join(missing_keys)))
        raise RuntimeError('; '.join(error_parts))

    if missing_keys:
        logger.warning('Missing keys while loading %s: %s', filename, ', '.join(missing_keys))
    if unexpected_keys:
        logger.warning('Unexpected keys while loading %s: %s', filename, ', '.join(unexpected_keys))

    return checkpoint
