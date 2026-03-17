import torch
from torch import nn


class VisualGuidedTokenReweighting(nn.Module):
    """Visual-guided token calibration before PWAM.

    Args:
        text_dim: Token feature dimension Ct.
        visual_dim: Global visual feature dimension Cv.
        hidden_dim: Hidden dimension used by the gate MLP.
        lambda_init: Initial residual scaling factor.
        use_layernorm: Whether to normalize text and visual projections before gating.
        gate_type: Only ``token_scalar`` is supported in v1.
    """

    def __init__(
        self,
        text_dim,
        visual_dim,
        hidden_dim=None,
        lambda_init=0.0,
        use_layernorm=True,
        gate_type='token_scalar',
    ):
        super().__init__()
        if gate_type != 'token_scalar':
            raise ValueError(f"Unsupported VGTR gate type '{gate_type}'. Only 'token_scalar' is supported.")

        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim or text_dim
        self.use_layernorm = use_layernorm
        self.gate_type = gate_type

        self.visual_proj = nn.Linear(visual_dim, text_dim)
        self.text_proj = nn.Linear(text_dim, self.hidden_dim)
        self.visual_context_proj = nn.Linear(text_dim, self.hidden_dim)
        self.gate_proj = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.GELU()
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init)))

        if self.use_layernorm:
            self.text_norm = nn.LayerNorm(text_dim)
            self.visual_norm = nn.LayerNorm(text_dim)
        else:
            self.text_norm = nn.Identity()
            self.visual_norm = nn.Identity()

        self.last_debug = {}

    def forward(self, text_feats, global_visual_feat, text_mask=None):
        if text_feats.ndim != 3:
            raise AssertionError(f'text_feats must be 3D [B, T, Ct], got shape {tuple(text_feats.shape)}')
        if global_visual_feat.ndim != 2:
            raise AssertionError(
                f'global_visual_feat must be 2D [B, Cv], got shape {tuple(global_visual_feat.shape)}'
            )
        if text_feats.size(0) != global_visual_feat.size(0):
            raise AssertionError('Batch size mismatch between text features and visual feature.')

        batch_size, token_count, text_dim = text_feats.shape
        if text_dim != self.text_dim:
            raise AssertionError(f'Expected text dim {self.text_dim}, got {text_dim}')

        gv_proj = self.visual_proj(global_visual_feat)  # [B, Ct]
        gv_expand = gv_proj.unsqueeze(1).expand(-1, token_count, -1)  # [B, T, Ct]

        text_norm = self.text_norm(text_feats)
        gv_norm = self.visual_norm(gv_expand)

        hidden = self.activation(self.text_proj(text_norm) + self.visual_context_proj(gv_norm))
        gate = torch.sigmoid(self.gate_proj(hidden))  # [B, T, 1]

        masked_fraction = 0.0
        if text_mask is not None:
            if text_mask.ndim == 3 and text_mask.size(-1) == 1:
                text_mask = text_mask.squeeze(-1)
            if text_mask.ndim != 2:
                raise AssertionError(f'text_mask must be 2D [B, T], got shape {tuple(text_mask.shape)}')
            if text_mask.shape != (batch_size, token_count):
                raise AssertionError(
                    f'text_mask shape {tuple(text_mask.shape)} does not match token shape {(batch_size, token_count)}'
                )
            text_mask = text_mask.to(dtype=text_feats.dtype)
            gate = gate * text_mask.unsqueeze(-1)
            masked_fraction = float((text_mask.numel() - text_mask.sum()).item() / max(text_mask.numel(), 1))

        reweighted_text_feats = text_feats + self.lambda_param * gate * text_feats
        if reweighted_text_feats.shape != text_feats.shape:
            raise AssertionError('VGTR output shape must match input text feature shape.')

        self.last_debug = {
            'text_mean': float(text_feats.mean().item()),
            'text_std': float(text_feats.std(unbiased=False).item()),
            'global_visual_mean': float(global_visual_feat.mean().item()),
            'global_visual_std': float(global_visual_feat.std(unbiased=False).item()),
            'gate_mean': float(gate.mean().item()),
            'gate_std': float(gate.std(unbiased=False).item()),
            'gate_min': float(gate.min().item()),
            'gate_max': float(gate.max().item()),
            'lambda_param': float(self.lambda_param.item()),
            'masked_fraction': masked_fraction,
        }

        return reweighted_text_feats
