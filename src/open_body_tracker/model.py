"""The network: a DINOv2 ViT-B/14 backbone with an RTMW-style SimCC head and a metric 3D head.

Input: a 256x192 (height x width) person crop, ImageNet-normalized RGB.
Outputs (logits):
  x (N, 23, 384), y (N, 23, 512)   2D SimCC classification, 2 bins per crop pixel
  X, Y, Z (N, 23, 300)              root-relative metric coordinates, 1 cm bins over [-1.5 m, 1.5 m]

The head follows RTMWHead / RTMCCBlock from MMPose (Apache-2.0, OpenMMLab), rewritten in plain PyTorch
with the same parameter names, so no MMPose or MMCV install is needed. See NOTICE.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .keypoints import NUM_KEYPOINTS

CROP_HEIGHT, CROP_WIDTH = 256, 192
GRID = (18, 14)            # ViT patch grid; the crop is resized to 252x196 inside the model
PATCH, DEPTH, NATIVE_GRID = 14, 12, 37
WIDTH, HEADS = 768, 12     # ViT-B/14
HEAD_CHANNELS = 768
SIMCC_SPLIT = 2.0
METRIC_RANGE, METRIC_BINS = 1.5, 300


# ---------------------------------------------------------------- backbone (DINOv2 ViT-B/14)

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.dim // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, self.dim))


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.attn, self.ls1 = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads), LayerScale(dim)
        self.norm2, self.mlp, self.ls2 = nn.LayerNorm(dim, eps=1e-6), Mlp(dim), LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, PATCH, PATCH)


class ViT(nn.Module):
    """DINOv2 ViT with the publisher's parameter names and a fixed 18x14 patch grid."""

    def __init__(self, dim=WIDTH, heads=HEADS, grid=GRID, checkpointing=False):
        super().__init__()
        self.grid, self.dim, self.checkpointing = grid, dim, checkpointing
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid[0] * grid[1], dim))
        self.patch_embed = PatchEmbed(dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(DEPTH))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def load_dinov2(self, state):
        """Load the official DINOv2 checkpoint (37x37 position grid), resizing positions to this grid."""
        state = dict(state)
        state.pop("mask_token", None)
        native = state["pos_embed"]
        patches = native[:, 1:].reshape(1, NATIVE_GRID, NATIVE_GRID, self.dim).permute(0, 3, 1, 2)
        patches = F.interpolate(patches, size=self.grid, mode="bicubic", align_corners=False)
        state["pos_embed"] = torch.cat((native[:, :1], patches.permute(0, 2, 3, 1).reshape(1, -1, self.dim)), 1)
        self.load_state_dict(state, strict=True)

    def forward(self, image):
        rows, cols = self.grid
        x = self.patch_embed.proj(image).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), 1) + self.pos_embed
        for block in self.blocks:
            if self.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)[:, 1:].transpose(1, 2).reshape(-1, self.dim, rows, cols)


class Neck(nn.Module):
    """Fine map: tokens projected to 384 channels (18x14). Coarse map: stride-2 conv to 768 channels (9x7)."""

    def __init__(self, dim=WIDTH):
        super().__init__()
        self.fine = nn.Sequential(nn.Conv2d(dim, HEAD_CHANNELS // 2, 1, bias=False), nn.BatchNorm2d(HEAD_CHANNELS // 2), nn.SiLU())
        self.conv = nn.Conv2d(dim, HEAD_CHANNELS, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(HEAD_CHANNELS)

    def forward(self, tokens):
        return self.fine(tokens), F.silu(self.bn(self.conv(tokens)))


# ---------------------------------------------------------------- head (RTMW-style, from MMPose)

class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, kernel):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, 1, kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")  # as MMCV ConvModule

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale, self.eps = dim ** -0.5, eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class GAU(nn.Module):
    """Gated attention unit (RTMCCBlock, self-attention, SiLU, no relative bias or rotary encoding)."""

    def __init__(self, dim=256, s=128, expansion=2):
        super().__init__()
        self.e, self.s = dim * expansion, s
        # Same construction order as MMPose RTMCCBlock, so a seeded initialisation matches the release run.
        self.o = nn.Linear(self.e, dim, bias=False)
        self.uv = nn.Linear(dim, 2 * self.e + s, bias=False)
        self.gamma = nn.Parameter(torch.rand(2, s))
        self.beta = nn.Parameter(torch.rand(2, s))
        self.ln = ScaleNorm(dim)
        nn.init.xavier_uniform_(self.uv.weight)
        self.res_scale = Scale(dim)

    def forward(self, x):
        u, v, base = torch.split(F.silu(self.uv(self.ln(x))), [self.e, self.e, self.s], dim=2)
        base = base.unsqueeze(2) * self.gamma + self.beta
        q, k = base[:, :, 0], base[:, :, 1]
        kernel = torch.square(F.relu(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.s)))
        return self.res_scale(x) + self.o(u * torch.bmm(kernel, v))


class Head(nn.Module):
    """keypoints=17, metric=False is the 2D model of training stage 1; the released model is 23 points with metric 3D."""

    def __init__(self, keypoints=NUM_KEYPOINTS, metric=True, hidden=256, kernel=7):
        super().__init__()
        coarse = (GRID[0] // 2) * (GRID[1] // 2)  # 9x7
        c = HEAD_CHANNELS
        # Same construction order as MMPose RTMWHead (matters only for the seeded initialisation).
        self.ps = nn.PixelShuffle(2)
        self.conv_dec = ConvBNReLU(c // 4, c // 4, kernel)
        self.final_layer = ConvBNReLU(c, keypoints, kernel)
        self.final_layer2 = ConvBNReLU(c // 2 + c // 4, keypoints, kernel)
        self.mlp = nn.Sequential(ScaleNorm(coarse), nn.Linear(coarse, hidden // 2, bias=False))
        self.mlp2 = nn.Sequential(ScaleNorm(coarse * 4), nn.Linear(coarse * 4, hidden // 2, bias=False))
        self.gau = GAU(hidden)
        self.cls_x = nn.Linear(hidden, int(CROP_WIDTH * SIMCC_SPLIT), bias=False)
        self.cls_y = nn.Linear(hidden, int(CROP_HEIGHT * SIMCC_SPLIT), bias=False)
        self.metric = metric
        if metric:
            self.cls_X = nn.Linear(hidden, METRIC_BINS, bias=False)
            self.cls_Y = nn.Linear(hidden, METRIC_BINS, bias=False)
            self.cls_Z = nn.Linear(hidden, METRIC_BINS, bias=False)

    def init_weights(self):
        """MMPose RTMWHead initialisation, in MMEngine's order (the same seed gives the same weights as the release run)."""
        for kind, init in ((nn.Conv2d, lambda m: nn.init.normal_(m.weight, 0, 0.001)),
                           (nn.BatchNorm2d, lambda m: (nn.init.ones_(m.weight), nn.init.zeros_(m.bias))),
                           (nn.Linear, lambda m: nn.init.normal_(m.weight, 0, 0.01))):
            for module in self.modules():
                if isinstance(module, kind):
                    init(module)
        # MMEngine then calls each ConvModule's own init_weights, which re-initialises its conv (Kaiming, fan-out).
        for module in self.modules():
            if isinstance(module, ConvBNReLU):
                nn.init.kaiming_normal_(module.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, feats):
        fine, coarse = feats
        tokens_coarse = self.mlp(torch.flatten(self.final_layer(coarse), 2))
        fine = torch.cat([self.conv_dec(self.ps(coarse)), fine], dim=1)
        tokens_fine = self.mlp2(torch.flatten(self.final_layer2(fine), 2))
        tokens = self.gau(torch.cat([tokens_coarse, tokens_fine], dim=2))
        layers = (self.cls_x, self.cls_y, self.cls_X, self.cls_Y, self.cls_Z) if self.metric else (self.cls_x, self.cls_y)
        return tuple(layer(tokens) for layer in layers)


class BodyModel(nn.Module):
    def __init__(self, keypoints=NUM_KEYPOINTS, metric=True, checkpointing=False):
        super().__init__()
        self.backbone = ViT(checkpointing=checkpointing)
        self.neck = Neck()
        self.head = Head(keypoints, metric)

    def forward(self, image):
        """image: (N, 3, 256, 192) normalized RGB -> (x, y, X, Y, Z) logits ((x, y) for the 2D-only model)."""
        image = F.interpolate(image, size=(GRID[0] * PATCH, GRID[1] * PATCH), mode="bilinear", align_corners=False)
        return self.head(self.neck(self.backbone(image)))


def metric_centres(device=None):
    return torch.linspace(-METRIC_RANGE, METRIC_RANGE, METRIC_BINS, device=device)


def decode(outputs):
    """Logits -> crop-pixel 2D (N,23,2), 2D confidence (N,23) and root-relative metres (N,23,3)."""
    x, y, X, Y, Z = outputs
    sx, ix = x.max(-1)
    sy, iy = y.max(-1)
    xy = torch.stack([ix, iy], -1).float() / SIMCC_SPLIT
    centres = metric_centres(x.device)
    xyz = torch.stack([(torch.softmax(v, -1) * centres).sum(-1) for v in (X, Y, Z)], -1)
    return xy, torch.minimum(sx, sy), xyz
