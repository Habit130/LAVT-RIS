import torch
import torch.nn as nn
from .mask_predictor import SimpleDecoding
from .backbone import MultiModalSwinTransformer
from ._utils import LAVT, LAVTOne

__all__ = ['lavt', 'lavt_one']


def _resolve_module_args(args):
    align_module = getattr(args, 'align_module', 'none')
    gate_module = getattr(args, 'gate_module', 'none')
    hlg_stages = tuple(getattr(args, 'hlg_stages', [3, 4]))

    if gate_module in ('lg', 'hlg') and align_module == 'none':
        raise ValueError('gate_module={} requires align_module to be one of plain/pwam/spam/hapwam'.format(gate_module))
    if gate_module == 'hlg':
        invalid_hlg_stages = [stage for stage in hlg_stages if stage not in (3, 4)]
        if invalid_hlg_stages:
            raise ValueError('gate_module=hlg only supports stages 3 and 4, got {}'.format(invalid_hlg_stages))

    return {
        'align_module': align_module,
        'gate_module': gate_module,
        'text_hidden_size': getattr(args, 'text_hidden_size', 768),
        'hapwam_hidden_dim': getattr(args, 'hapwam_hidden_dim', 256),
        'hapwam_fusion_hidden_dim': getattr(args, 'hapwam_fusion_hidden_dim', 256),
        'hapwam_dropout': getattr(args, 'hapwam_dropout', 0.1),
        'hlg_hidden_channels': getattr(args, 'hlg_hidden_channels', None),
        'hlg_stages': hlg_stages,
    }


# LAVT
def _segm_lavt(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    module_args = _resolve_module_args(args)

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    elif module_args['align_module'] in ('spam', 'hapwam'):
        mha = [1, 1, 4, 8]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop,
                                         align_module=module_args['align_module'],
                                         gate_module=module_args['gate_module'],
                                         text_hidden_size=module_args['text_hidden_size'],
                                         hapwam_hidden_dim=module_args['hapwam_hidden_dim'],
                                         hapwam_fusion_hidden_dim=module_args['hapwam_fusion_hidden_dim'],
                                         hapwam_dropout=module_args['hapwam_dropout'],
                                         hlg_hidden_channels=module_args['hlg_hidden_channels'],
                                         hlg_stages=module_args['hlg_stages']
                                         )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    model_map = [SimpleDecoding, LAVT]

    classifier = model_map[0](8*embed_dim)
    base_model = model_map[1]

    model = base_model(backbone, classifier)
    return model


def _load_model_lavt(pretrained, args):
    model = _segm_lavt(pretrained, args)
    return model


def lavt(pretrained='', args=None):
    return _load_model_lavt(pretrained, args)


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
def _segm_lavt_one(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    module_args = _resolve_module_args(args)

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    elif module_args['align_module'] in ('spam', 'hapwam'):
        mha = [1, 1, 4, 8]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop,
                                         align_module=module_args['align_module'],
                                         gate_module=module_args['gate_module'],
                                         text_hidden_size=module_args['text_hidden_size'],
                                         hapwam_hidden_dim=module_args['hapwam_hidden_dim'],
                                         hapwam_fusion_hidden_dim=module_args['hapwam_fusion_hidden_dim'],
                                         hapwam_dropout=module_args['hapwam_dropout'],
                                         hlg_hidden_channels=module_args['hlg_hidden_channels'],
                                         hlg_stages=module_args['hlg_stages']
                                         )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    model_map = [SimpleDecoding, LAVTOne]

    classifier = model_map[0](8*embed_dim)
    base_model = model_map[1]

    model = base_model(backbone, classifier, args)
    return model


def _load_model_lavt_one(pretrained, args):
    model = _segm_lavt_one(pretrained, args)
    return model


def lavt_one(pretrained='', args=None):
    return _load_model_lavt_one(pretrained, args)
