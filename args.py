import argparse

from text_encoder import DEFAULT_MAX_TEXT_TOKENS, DEFAULT_TEXT_ENCODER_NAME


ABLATION_CONFIGS = {
    'base': ('plain', 'none', [3, 4]),
    'spam_only': ('spam', 'none', [3, 4]),
    'hlg_only': ('plain', 'hlg', [3, 4]),
    'ours': ('spam', 'hlg', [3, 4]),
    'lavt_style_baseline': ('pwam', 'lg', [3, 4]),
}


def apply_ablation_config(args):
    ablation_config = getattr(args, 'ablation_config', '')
    if not ablation_config:
        return args

    align_module, gate_module, hlg_stages = ABLATION_CONFIGS[ablation_config]
    args.align_module = align_module
    args.gate_module = gate_module
    args.hlg_stages = list(hlg_stages)
    print('Resolved ablation_config [{}] to align_module={}, gate_module={}, hlg_stages={}'.format(
        ablation_config, args.align_module, args.gate_module, args.hlg_stages))
    return args


def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--text_tokenizer_name', default='',
                        help='tokenizer name or local path; defaults to --text_encoder_name when omitted')
    parser.add_argument('--text_encoder_name', default=DEFAULT_TEXT_ENCODER_NAME,
                        help='pre-trained text encoder name or local path')
    parser.add_argument('--max_text_tokens', default=DEFAULT_MAX_TEXT_TOKENS, type=int,
                        help='maximum tokenized text length after truncation/padding')
    parser.add_argument('--bert_tokenizer', default=None,
                        help='deprecated alias of --text_tokenizer_name')
    parser.add_argument('--ck_bert', default=None,
                        help='deprecated alias of --text_encoder_name')
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, refcocog, or plantseg')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0',
                        help='device for testing or single-GPU training')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--ablation_config', default='base',
                        choices=[''] + sorted(ABLATION_CONFIGS.keys()),
                        help='clean RIS ablation preset; pass an empty string to use raw align/gate switches directly')
    parser.add_argument('--align_module', default='spam', choices=['none', 'plain', 'pwam', 'spam', 'hapwam'],
                        help='raw stage-level language alignment module; final ablations should use --ablation_config')
    parser.add_argument('--fusion_drop', default=0.0, type=float,
                        help='dropout rate for plain/PWAM/SPAM fusion modules')
    parser.add_argument('--gate_module', default='hlg', choices=['none', 'lg', 'hlg'],
                        help='raw gate module after alignment; lg is the legacy res_gate path')
    parser.add_argument('--hapwam_hidden_dim', default=256, type=int,
                        help='compatibility alias for the hidden dimension of the SPAM token reweighting branch')
    parser.add_argument('--hapwam_fusion_hidden_dim', default=256, type=int,
                        help='compatibility alias for the hidden dimension of the SPAM anomaly modulation branch')
    parser.add_argument('--hapwam_dropout', default=0.1, type=float,
                        help='compatibility alias retained for older SPAM/HAPWAM experiment configs')
    parser.add_argument('--hlg_aux_loss_weight', default=0.05, type=float,
                        help='weight for the summed HLG auxiliary loss')
    parser.add_argument('--hlg_false_healthy_weight', default=0.5, type=float,
                        help='weight for false-healthy suppression inside each HLG stage loss')
    parser.add_argument('--hlg_hidden_channels', default=None, type=int,
                        help='hidden channels for HLG; defaults to the stage channel dimension when omitted')
    parser.add_argument('--hlg_stages', default=[3, 4], nargs='+', type=int,
                        help='1-based stage ids that use HLG when gate_module=hlg; defaults to stages 3 and 4')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument("--local_rank", default=-1, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='lavt', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--plantseg_root', default='../plantseg',
                        help='plantseg dataset root directory')
    parser.add_argument('--plantseg_caption_index', default=3, type=int,
                        help='0-based caption index used for plantseg samples')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='local path or HTTPS URL to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--save_pred_dir', default='',
                        help='directory for saving predicted masks during testing; testing metrics are computed separately via eval_ris_metrics.py')
    parser.add_argument('--save_mask_dir', default='',
                        help='deprecated alias of --save_pred_dir')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    return parser


def validate_args(args):
    args = apply_ablation_config(args)

    if args.ck_bert:
        if args.text_encoder_name == DEFAULT_TEXT_ENCODER_NAME:
            print('Deprecated argument --ck_bert detected; mapping it to --text_encoder_name.')
            args.text_encoder_name = args.ck_bert
        elif args.ck_bert != args.text_encoder_name:
            print('Ignoring deprecated --ck_bert because --text_encoder_name is already set to [{}].'.format(
                args.text_encoder_name))

    if args.bert_tokenizer:
        if not args.text_tokenizer_name:
            print('Deprecated argument --bert_tokenizer detected; mapping it to --text_tokenizer_name.')
            args.text_tokenizer_name = args.bert_tokenizer
        elif args.bert_tokenizer != args.text_tokenizer_name:
            print('Ignoring deprecated --bert_tokenizer because --text_tokenizer_name is already set to [{}].'.format(
                args.text_tokenizer_name))

    if not args.text_tokenizer_name:
        args.text_tokenizer_name = args.text_encoder_name
    if args.save_mask_dir:
        if not args.save_pred_dir:
            print('Deprecated argument --save_mask_dir detected; mapping it to --save_pred_dir.')
            args.save_pred_dir = args.save_mask_dir
        elif args.save_mask_dir != args.save_pred_dir:
            print('Ignoring deprecated --save_mask_dir because --save_pred_dir is already set to [{}].'.format(
                args.save_pred_dir))
    if args.max_text_tokens < 1:
        raise ValueError('--max_text_tokens must be >= 1')
    if args.hapwam_hidden_dim < 1:
        raise ValueError('--hapwam_hidden_dim must be >= 1')
    if args.hapwam_fusion_hidden_dim < 1:
        raise ValueError('--hapwam_fusion_hidden_dim must be >= 1')

    align_module = getattr(args, 'align_module', 'none')
    gate_module = getattr(args, 'gate_module', 'none')
    hlg_stages = tuple(getattr(args, 'hlg_stages', [3, 4]))

    if gate_module in ('lg', 'hlg') and align_module == 'none':
        raise ValueError('gate_module={} requires align_module to be one of plain/pwam/spam/hapwam'.format(gate_module))

    if gate_module == 'hlg':
        invalid_hlg_stages = [stage for stage in hlg_stages if stage not in (3, 4)]
        if invalid_hlg_stages:
            raise ValueError('gate_module=hlg only supports stages 3 and 4, got {}'.format(invalid_hlg_stages))
        if not hlg_stages:
            raise ValueError('gate_module=hlg requires at least one stage id in --hlg_stages')

    return args


if __name__ == "__main__":
    parser = get_parser()
    args_dict = validate_args(parser.parse_args())
