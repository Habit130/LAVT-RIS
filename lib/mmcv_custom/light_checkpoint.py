import os
import os.path as osp
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.hub import load_state_dict_from_url


def mkdir_or_exist(path):
    if path:
        os.makedirs(path, exist_ok=True)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def is_module_wrapper(module):
    return isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel))


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(target_module, prefix=''):
        if is_module_wrapper(target_module):
            target_module = target_module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        target_module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                            all_missing_keys, unexpected_keys, err_msg)
        for name, child in target_module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))

    if strict and err_msg:
        rank, _ = get_dist_info()
        if rank == 0:
            err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
            message = '\n'.join(err_msg)
            if logger is not None:
                logger.warning(message)
            raise RuntimeError(message)


def _load_checkpoint(filename, map_location=None):
    if filename.startswith(('http://', 'https://')):
        return load_state_dict_from_url(filename, map_location=map_location, progress=True)
    if not osp.isfile(filename):
        raise IOError('{} is not a checkpoint file'.format(filename))
    return torch.load(filename, map_location=map_location)


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    first_key = list(state_dict.keys())[0]
    if first_key.startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        first_key = list(state_dict.keys())[0]
    if first_key.startswith('backbone.'):
        print('Start stripping upper net pre-fix and loading backbone weights to our swin encoder')
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
        first_key = list(state_dict.keys())[0]
    if first_key.startswith('encoder.'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        n1, l1, c1 = absolute_pos_embed.size()
        n2, c2, h, w = model.absolute_pos_embed.size()
        if n1 == n2 and c1 == c2 and l1 == h * w:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(n2, h, w, c2).permute(0, 3, 1, 2)
        elif logger is not None:
            logger.warning('Error in loading absolute_pos_embed, pass')

    relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
    model_state_dict = model.state_dict()
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model_state_dict[table_key]
        l1, n_h1 = table_pretrained.size()
        l2, n_h2 = table_current.size()
        if n_h1 != n_h2:
            if logger is not None:
                logger.warning('Error in loading %s, pass', table_key)
            continue
        if l1 != l2:
            s1 = int(l1 ** 0.5)
            s2 = int(l2 ** 0.5)
            resized = F.interpolate(
                table_pretrained.permute(1, 0).view(1, n_h1, s1, s1),
                size=(s2, s2),
                mode='bicubic'
            )
            state_dict[table_key] = resized.view(n_h2, l2).permute(1, 0)

    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(state_dict):
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def _save_to_state_dict(module, destination, prefix, keep_vars):
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    if is_module_wrapper(module):
        module = module.module

    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = dict(version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, destination._metadata[prefix[:-1]])
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(model, filename, optimizer=None, meta=None):
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(type(meta)))
    meta.update(time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    mkdir_or_exist(osp.dirname(filename))
    with open(filename, 'wb') as handle:
        torch.save(checkpoint, handle)
        handle.flush()
