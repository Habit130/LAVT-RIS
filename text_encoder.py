from pathlib import Path


DEFAULT_TEXT_ENCODER_NAME = 'microsoft/deberta-v3-base'
DEFAULT_MAX_TEXT_TOKENS = 64
LEGACY_BERT_MODEL_KEY = 'bert_model'
TEXT_ENCODER_MODEL_KEY = 'text_encoder'


def _import_transformers_attr(module_name, attr_name):
    try:
        module = __import__('transformers', fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception as exc:
        raise RuntimeError(
            'Failed to import transformers component [{}] for text encoder setup. '
            'Make sure the active environment installs `transformers`, `sentencepiece`, and uses `numpy<2`. '
            'Recommended fix on the server: `conda env update -f environment.server.yml --prune`.'
            .format(attr_name)
        ) from exc


def _normalize_model_name(name):
    if not name:
        return ''
    return Path(str(name)).name.lower()


def prepare_text_encoder_args(args):
    AutoConfig = _import_transformers_attr('transformers', 'AutoConfig')

    config = AutoConfig.from_pretrained(args.text_encoder_name)
    args.text_hidden_size = get_text_hidden_size(config)
    args.text_encoder_model_type = getattr(config, 'model_type', '')
    if not getattr(args, 'text_tokenizer_name', ''):
        args.text_tokenizer_name = args.text_encoder_name
    return config


def build_text_tokenizer(args):
    tokenizer_name = args.text_tokenizer_name or args.text_encoder_name
    model_type = getattr(args, 'text_encoder_model_type', '')

    if model_type == 'deberta-v2':
        try:
            DebertaV2TokenizerFast = _import_transformers_attr('transformers', 'DebertaV2TokenizerFast')

            return DebertaV2TokenizerFast.from_pretrained(tokenizer_name)
        except (ImportError, OSError, ValueError):
            DebertaV2Tokenizer = _import_transformers_attr('transformers', 'DebertaV2Tokenizer')

            return DebertaV2Tokenizer.from_pretrained(tokenizer_name)

    if model_type == 'bert':
        try:
            BertTokenizerFast = _import_transformers_attr('transformers', 'BertTokenizerFast')

            return BertTokenizerFast.from_pretrained(tokenizer_name)
        except (ImportError, OSError, ValueError):
            BertTokenizer = _import_transformers_attr('transformers', 'BertTokenizer')

            return BertTokenizer.from_pretrained(tokenizer_name)

    AutoTokenizer = _import_transformers_attr('transformers', 'AutoTokenizer')

    return AutoTokenizer.from_pretrained(tokenizer_name)


def build_text_encoder(args, config=None):
    AutoConfig = _import_transformers_attr('transformers', 'AutoConfig')

    if config is None:
        config = AutoConfig.from_pretrained(args.text_encoder_name)
    if getattr(config, 'model_type', '') == 'deberta-v2':
        DebertaV2Model = _import_transformers_attr('transformers', 'DebertaV2Model')

        model = DebertaV2Model.from_pretrained(args.text_encoder_name, config=config)
    elif getattr(config, 'model_type', '') == 'bert':
        BertModel = _import_transformers_attr('transformers', 'BertModel')

        model = BertModel.from_pretrained(args.text_encoder_name, config=config)
    else:
        AutoModel = _import_transformers_attr('transformers', 'AutoModel')

        model = AutoModel.from_pretrained(args.text_encoder_name, config=config)
    if hasattr(model, 'pooler'):
        model.pooler = None
    return model


def tokenize_text(tokenizer, text, max_length):
    encoded = tokenizer(text,
                        add_special_tokens=True,
                        truncation=True,
                        padding='max_length',
                        max_length=max_length,
                        return_attention_mask=True,
                        return_tensors='pt')
    return encoded['input_ids'], encoded['attention_mask']


def encode_text(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    return outputs[0]


def get_text_hidden_size(source):
    config = getattr(source, 'config', source)
    hidden_size = getattr(config, 'hidden_size', None)
    if hidden_size is None:
        raise ValueError('Cannot infer text hidden size from {}'.format(type(source).__name__))
    return hidden_size


def get_text_encoder_layers(model):
    candidate_paths = (
        ('encoder', 'layer'),
        ('deberta', 'encoder', 'layer'),
        ('bert', 'encoder', 'layer'),
        ('roberta', 'encoder', 'layer'),
    )

    for path in candidate_paths:
        current = model
        for attr in path:
            if not hasattr(current, attr):
                current = None
                break
            current = getattr(current, attr)
        if current is not None:
            return list(current)

    raise AttributeError('Unsupported text encoder layer layout: {}'.format(type(model).__name__))


def is_legacy_bert_encoder(args):
    model_type = getattr(args, 'text_encoder_model_type', '')
    if model_type == 'bert':
        return True
    return _normalize_model_name(getattr(args, 'text_encoder_name', '')).startswith('bert')


def get_checkpoint_text_encoder_state(checkpoint, args):
    if TEXT_ENCODER_MODEL_KEY in checkpoint:
        return checkpoint[TEXT_ENCODER_MODEL_KEY], TEXT_ENCODER_MODEL_KEY

    if LEGACY_BERT_MODEL_KEY in checkpoint:
        if is_legacy_bert_encoder(args):
            print('Loading legacy checkpoint key [{}] for text encoder.'.format(LEGACY_BERT_MODEL_KEY))
            return checkpoint[LEGACY_BERT_MODEL_KEY], LEGACY_BERT_MODEL_KEY
        print('Skipping legacy checkpoint key [{}] because current text encoder is [{}].'.format(
            LEGACY_BERT_MODEL_KEY, args.text_encoder_name))

    return None, None
