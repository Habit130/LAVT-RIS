import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()
        self.last_boundary_logits = None

        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        return self.conv1_1(x)


class LesionAwareFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LesionAwareFusionBlock, self).__init__()
        branch_channels = max(out_channels // 2, 32)

        self.local_branch = ConvBNReLU(in_channels, branch_channels, kernel_size=3, padding=1)
        self.context_branch = ConvBNReLU(
            in_channels,
            branch_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
        )
        self.boundary_branch = nn.Sequential(
            ConvBNReLU(in_channels, branch_channels, kernel_size=1),
            ConvBNReLU(branch_channels, branch_channels, kernel_size=3, padding=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        local_feat = self.local_branch(x)
        context_feat = self.context_branch(x)
        boundary_feat = self.boundary_branch(x)
        x = torch.cat([local_feat, context_feat, boundary_feat], dim=1)
        return self.fuse(x)


class DiseaseAwareRefinementHead(nn.Module):
    def __init__(
        self,
        c4_dims,
        factor=2,
        use_boundary_refine=False,
        boundary_loss_weight=0.0,
        boundary_alpha=0.1,
    ):
        super(DiseaseAwareRefinementHead, self).__init__()
        self.last_boundary_logits = None
        self.use_boundary_refine = use_boundary_refine
        self.boundary_loss_weight = boundary_loss_weight
        self.enable_boundary_branch = use_boundary_refine or boundary_loss_weight > 0.0

        hidden_size = c4_dims // factor
        c4_size = c4_dims
        c3_size = c4_dims // (factor ** 1)
        c2_size = c4_dims // (factor ** 2)
        c1_size = c4_dims // (factor ** 3)

        self.proj_c4 = ConvBNReLU(c4_size, hidden_size, kernel_size=1)
        self.fuse_c3 = LesionAwareFusionBlock(hidden_size + c3_size, hidden_size)
        self.fuse_c2 = LesionAwareFusionBlock(hidden_size + c2_size, hidden_size)
        self.fuse_c1 = LesionAwareFusionBlock(hidden_size + c1_size, hidden_size)
        self.mask_head = nn.Conv2d(hidden_size, 2, kernel_size=1)

        if self.enable_boundary_branch:
            boundary_hidden = max(hidden_size // 2, 32)
            self.boundary_head = nn.Sequential(
                ConvBNReLU(hidden_size, boundary_hidden, kernel_size=3, padding=1),
                nn.Conv2d(boundary_hidden, 1, kernel_size=1),
            )
            self.boundary_alpha = nn.Parameter(torch.tensor(float(boundary_alpha)))
        else:
            self.boundary_head = None
            self.register_parameter('boundary_alpha', None)

    @staticmethod
    def _resize_like(x, ref):
        if x.size(-2) != ref.size(-2) or x.size(-1) != ref.size(-1):
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=True)
        return x

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        z4 = self.proj_c4(x_c4)
        z3 = self.fuse_c3(torch.cat([self._resize_like(z4, x_c3), x_c3], dim=1))
        z2 = self.fuse_c2(torch.cat([self._resize_like(z3, x_c2), x_c2], dim=1))
        z1 = self.fuse_c1(torch.cat([self._resize_like(z2, x_c1), x_c1], dim=1))

        main_logits = self.mask_head(z1)
        self.last_boundary_logits = None

        if not self.enable_boundary_branch:
            return main_logits

        boundary_logits = self.boundary_head(z1)
        self.last_boundary_logits = boundary_logits
        if self.use_boundary_refine:
            fg_logits = main_logits[:, 1:2] + self.boundary_alpha * boundary_logits
            main_logits = torch.cat([main_logits[:, 0:1], fg_logits], dim=1)

        return main_logits
