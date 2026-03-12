import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data

from bert.tokenization_bert import BertTokenizer


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = ['background', 'foreground']
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.eval_mode = eval_mode
        self.max_tokens = args.text_max_tokens
        self.dataset_root = Path(args.dataset_root).expanduser().resolve()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        split_files = {
            'train': args.train_json,
            'test': args.test_json,
        }
        if self.split not in split_files:
            raise ValueError(
                f"Unsupported split '{self.split}'. Available splits: {sorted(split_files)}"
            )

        annotations_path = self.dataset_root / split_files[self.split]
        if not annotations_path.is_file():
            raise FileNotFoundError(f'Annotation file not found: {annotations_path}')

        with annotations_path.open('r', encoding='utf-8') as handle:
            samples = json.load(handle)

        if not isinstance(samples, list):
            raise ValueError(f'Expected a list of samples in {annotations_path}, got {type(samples).__name__}')

        self.samples = []
        self.input_ids = []
        self.attention_masks = []

        for sample in samples:
            image_path = self._resolve_path(sample.get('image'))
            mask_path = self._resolve_path(sample.get('mask'))
            captions = self._normalize_captions(sample)
            sample_id = sample.get('id', image_path.stem)

            self.samples.append({
                'id': sample_id,
                'image_path': image_path,
                'mask_path': mask_path,
                'captions': captions,
            })

            sentences_for_sample = []
            attentions_for_sample = []
            for caption in captions:
                token_ids = self.tokenizer.encode(text=caption, add_special_tokens=True)
                token_ids = token_ids[:self.max_tokens]

                padded_input_ids = [0] * self.max_tokens
                attention_mask = [0] * self.max_tokens
                padded_input_ids[:len(token_ids)] = token_ids
                attention_mask[:len(token_ids)] = [1] * len(token_ids)

                sentences_for_sample.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_sample.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_sample)
            self.attention_masks.append(attentions_for_sample)

    def _normalize_captions(self, sample):
        raw_captions = sample.get('caption')
        if raw_captions is None:
            raw_captions = sample.get('captions')
        if raw_captions is None:
            raw_captions = sample.get('sentence')
        if raw_captions is None:
            raw_captions = sample.get('sentences')
        if raw_captions is None:
            raise KeyError(f"Sample {sample.get('id', '<unknown>')} does not contain a caption field")

        if isinstance(raw_captions, str):
            captions = [raw_captions]
        elif isinstance(raw_captions, list):
            captions = [str(caption) for caption in raw_captions if str(caption).strip()]
        else:
            raise TypeError(
                f"Sample {sample.get('id', '<unknown>')} has unsupported caption type {type(raw_captions).__name__}"
            )

        if not captions:
            raise ValueError(f"Sample {sample.get('id', '<unknown>')} does not contain a usable caption")

        return captions

    def _resolve_path(self, path_value):
        if not path_value:
            raise KeyError(f"Split '{self.split}' contains a sample without image/mask path")

        path = Path(path_value)
        if not path.is_absolute():
            path = self.dataset_root / path
        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(f'File referenced by split {self.split} not found: {path}')

        return path

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample['image_path']).convert('RGB')
        mask = np.array(Image.open(sample['mask_path']))
        # Collapse all non-zero class ids into a single foreground class.
        mask = (mask > 0).astype(np.uint8)
        target = Image.fromarray(mask, mode='L')

        if self.image_transforms is not None:
            image, target = self.image_transforms(image, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.eval_mode:
            embeddings = []
            attentions = []
            for sentence_tensor, attention_tensor in zip(self.input_ids[index], self.attention_masks[index]):
                embeddings.append(sentence_tensor.unsqueeze(-1))
                attentions.append(attention_tensor.unsqueeze(-1))

            tensor_embeddings = torch.cat(embeddings, dim=-1)
            attention_mask = torch.cat(attentions, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return image, target, tensor_embeddings, attention_mask
