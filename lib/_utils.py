from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask, return_aux=False):
        input_shape = x.shape[-2:]
        if return_aux:
            features, aux = self.backbone(x, l_feats, l_mask, return_aux=True)
        else:
            features = self.backbone(x, l_feats, l_mask)
            aux = None
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        if return_aux:
            return x, aux
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
        from bert.modeling_bert import BertModel
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask, return_aux=False):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        if return_aux:
            features, aux = self.backbone(x, l_feats, l_mask, return_aux=True)
        else:
            features = self.backbone(x, l_feats, l_mask)
            aux = None
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        if return_aux:
            return x, aux
        return x


class LAVTOne(_LAVTOneSimpleDecode):
    pass
