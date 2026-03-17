from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel

from .vgtr import VisualGuidedTokenReweighting


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args=None):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.use_vgtr = getattr(args, 'use_vgtr', True)
        self.vgtr_source_stage = getattr(args, 'vgtr_source_stage', 4)
        self.vgtr_debug = getattr(args, 'vgtr_debug', False)
        self.last_vgtr_debug = None

        if self.use_vgtr:
            if self.vgtr_source_stage < 1 or self.vgtr_source_stage > len(self.backbone.num_features):
                raise ValueError(
                    f'vgtr_source_stage must be in [1, {len(self.backbone.num_features)}], '
                    f'got {self.vgtr_source_stage}'
                )
            text_dim = 768
            visual_dim = self.backbone.num_features[self.vgtr_source_stage - 1]
            self.vgtr = VisualGuidedTokenReweighting(
                text_dim=text_dim,
                visual_dim=visual_dim,
                hidden_dim=getattr(args, 'vgtr_hidden_dim', text_dim),
                lambda_init=getattr(args, 'vgtr_lambda_init', 0.0),
                use_layernorm=getattr(args, 'vgtr_use_layernorm', True),
                gate_type=getattr(args, 'vgtr_gate_type', 'token_scalar'),
            )
        else:
            self.vgtr = None

    def _apply_vgtr(self, image, text_feats, text_mask):
        if self.vgtr is None:
            self.last_vgtr_debug = None
            return text_feats

        visual_stages = self.backbone.forward_visual_features(image)
        source_feature = visual_stages[self.vgtr_source_stage - 1]
        global_visual_feat = source_feature.mean(dim=(2, 3))
        reweighted_text_feats = self.vgtr(text_feats, global_visual_feat, text_mask=text_mask)
        self.last_vgtr_debug = dict(self.vgtr.last_debug)

        if self.vgtr_debug:
            debug = self.last_vgtr_debug
            print(
                'VGTR '
                f'lambda={debug["lambda_param"]:.4f} '
                f'gate_mean={debug["gate_mean"]:.4f} '
                f'gate_std={debug["gate_std"]:.4f} '
                f'gate_min={debug["gate_min"]:.4f} '
                f'gate_max={debug["gate_max"]:.4f} '
                f'masked_fraction={debug["masked_fraction"]:.4f}'
            )

        return reweighted_text_feats

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        text_mask = l_mask.squeeze(-1) if l_mask.ndim == 3 else l_mask
        text_feats = l_feats.permute(0, 2, 1)
        text_feats = self._apply_vgtr(x, text_feats, text_mask=text_mask)
        l_feats = text_feats.permute(0, 2, 1)
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None
        self.use_vgtr = getattr(args, 'use_vgtr', True)
        self.vgtr_source_stage = getattr(args, 'vgtr_source_stage', 4)
        self.vgtr_debug = getattr(args, 'vgtr_debug', False)
        self.last_vgtr_debug = None

        if self.use_vgtr:
            if self.vgtr_source_stage < 1 or self.vgtr_source_stage > len(self.backbone.num_features):
                raise ValueError(
                    f'vgtr_source_stage must be in [1, {len(self.backbone.num_features)}], '
                    f'got {self.vgtr_source_stage}'
                )
            text_dim = self.text_encoder.config.hidden_size
            visual_dim = self.backbone.num_features[self.vgtr_source_stage - 1]
            self.vgtr = VisualGuidedTokenReweighting(
                text_dim=text_dim,
                visual_dim=visual_dim,
                hidden_dim=getattr(args, 'vgtr_hidden_dim', text_dim),
                lambda_init=getattr(args, 'vgtr_lambda_init', 0.0),
                use_layernorm=getattr(args, 'vgtr_use_layernorm', True),
                gate_type=getattr(args, 'vgtr_gate_type', 'token_scalar'),
            )
        else:
            self.vgtr = None

    def _apply_vgtr(self, image, text_feats, text_mask):
        if self.vgtr is None:
            self.last_vgtr_debug = None
            return text_feats

        visual_stages = self.backbone.forward_visual_features(image)
        source_feature = visual_stages[self.vgtr_source_stage - 1]
        global_visual_feat = source_feature.mean(dim=(2, 3))
        reweighted_text_feats = self.vgtr(text_feats, global_visual_feat, text_mask=text_mask)
        self.last_vgtr_debug = dict(self.vgtr.last_debug)

        if self.vgtr_debug:
            debug = self.last_vgtr_debug
            print(
                'VGTR '
                f'lambda={debug["lambda_param"]:.4f} '
                f'gate_mean={debug["gate_mean"]:.4f} '
                f'gate_std={debug["gate_std"]:.4f} '
                f'gate_min={debug["gate_min"]:.4f} '
                f'gate_max={debug["gate_max"]:.4f} '
                f'masked_fraction={debug["masked_fraction"]:.4f}'
            )

        return reweighted_text_feats

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        text_mask = l_mask
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (B, N_l, 768)
        l_feats = self._apply_vgtr(x, l_feats, text_mask=text_mask)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVTOne(_LAVTOneSimpleDecode):
    pass
