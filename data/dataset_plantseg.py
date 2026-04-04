import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from bert.tokenization_bert import BertTokenizer


class PlantSegDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):
        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.eval_mode = eval_mode
        self.max_tokens = 20
        self.caption_index = args.plantseg_caption_index
        self.root = Path(args.plantseg_root).expanduser().resolve()
        self.metadata_path = self.root / 'main.json'

        with self.metadata_path.open('r', encoding='utf-8') as handle:
            records = json.load(handle)

        self.samples = [record for record in records if record.get('split') == split]
        if not self.samples:
            raise ValueError('No plantseg samples found for split [{}] under {}'.format(split, self.metadata_path))

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.input_ids = []
        self.attention_masks = []

        for sample in self.samples:
            captions = sample.get('caption') or []
            if len(captions) <= self.caption_index:
                raise IndexError(
                    'Sample [{}] does not have caption index {}'.format(sample.get('id', '<unknown>'),
                                                                        self.caption_index)
                )

            sentence_raw = captions[self.caption_index]
            attention_mask = [0] * self.max_tokens
            padded_input_ids = [0] * self.max_tokens

            input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
            input_ids = input_ids[:self.max_tokens]

            padded_input_ids[:len(input_ids)] = input_ids
            attention_mask[:len(input_ids)] = [1] * len(input_ids)

            self.input_ids.append(torch.tensor(padded_input_ids).unsqueeze(0))
            self.attention_masks.append(torch.tensor(attention_mask).unsqueeze(0))

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.root / sample['image']
        mask_path = self.root / sample['mask']

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        mask_array = (np.array(mask) > 0).astype(np.uint8)
        target = Image.fromarray(mask_array, mode='P')

        if self.image_transforms is not None:
            image, target = self.image_transforms(image, target)

        sentence = self.input_ids[index]
        attention_mask = self.attention_masks[index]

        if self.eval_mode:
            sentence = sentence.unsqueeze(-1)
            attention_mask = attention_mask.unsqueeze(-1)

        return image, target, sentence, attention_mask

