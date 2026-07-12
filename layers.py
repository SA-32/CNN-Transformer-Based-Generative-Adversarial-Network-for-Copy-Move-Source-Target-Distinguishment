"""
Building blocks for CNN-T GAN (Zhang et al., "CNN-Transformer Based
Generative Adversarial Network for Copy-Move Source/Target
Distinguishment", IEEE TCSVT 2023).

This module implements, following Sec. III-B of the paper:
  * the CNN branch's bottleneck / conv-stage design
  * the transformer branch's multi-head self-attention + MLP blocks
  * the Feature Coupling Layers (down-sampling and up-sampling, Fig. 6)
  * the Pearson Correlation Layer (Eqs. 7-12)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CNN branch building blocks (Sec. III-B.2)
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    """
    1x1 down-projection conv -> 3x3 spatial conv -> 1x1 up-projection conv,
    each followed by BN + ReLU, with a residual connection between the
    block's input and output, as described in Sec. III-B.2:

        "Each bottleneck consists of a 1x1 down-projection convolutional
         layer, a 3x3 spatial convolutional layer, and a 1x1 up-projection
         convolutional layer, each convolutional layer is followed by a
         BN layer and a ReLU layer. A residual connection is added between
         the input and the output of the bottleneck."
    """

    def __init__(self, in_channels, out_channels, stride=1, reduction=4):
        super().__init__()
        mid_channels = max(out_channels // reduction, 1)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


class ConvStage(nn.Module):
    """
    One of the three convolutional blocks of the CNN branch (Sec. III-B.2):

      Conv block1: 7 bottlenecks, 64  -> 256  channels, spatial 64x64 (no downsampling)
      Conv block2: 8 bottlenecks, 256 -> 512  channels, spatial 64x64 -> 32x32
      Conv block3: 6 bottlenecks, 512 -> 1024 channels, spatial 32x32 -> 16x16

    The first bottleneck of a stage performs the channel expansion (and,
    for stage 2/3, the spatial downsampling); the remaining bottlenecks
    keep channel count and resolution fixed. The paper further groups the
    bottlenecks (after the very first one in Conv block1) into pairs, each
    pair corresponding to a "stage" that interacts with one transformer
    block through the feature coupling layers -- this pairing does not
    change the tensor shapes, only which sub-blocks the FCLs are attached
    to, which is handled at the Generator level.
    """

    def __init__(self, in_channels, out_channels, num_bottlenecks, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        blocks = [Bottleneck(in_channels, out_channels, stride=stride)]
        for _ in range(num_bottlenecks - 1):
            blocks.append(Bottleneck(out_channels, out_channels, stride=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Transformer branch building blocks (Sec. III-B.1)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention, Fig. 5 and Eqs. (5)-(6).

    N (num_heads) = 12, d_model = 768, d_k = d_v = d_model / N = 64, as
    specified in the paper.
    """

    def __init__(self, d_model=768, num_heads=12, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        # Eq. (5): Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        # Eq. (6): MultiHeads(Q, K, V) = Concat(head_1, ..., head_N) W^O
        return self.out_proj(out)


class MLP(nn.Module):
    """
    "An MLP layer consists of an up-projection fully connected layer, a
    Gaussian error leaky unit (GELU) layer, a down-projection layer, and a
    dropout layer." (Sec. III-B.1)
    """

    def __init__(self, d_model=768, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MHSA_MLP_Pair(nn.Module):
    """
    A single (MHSA, MLP) pair. "A layernorm layer is added before the MHSA
    blocks and the MLP layers ... Residual connections are added in both
    the self-attention layer and MLP layers." (Sec. III-B.1)
    """

    def __init__(self, d_model=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStage(nn.Module):
    """
    One of the three transformer blocks. Trans block1 and Trans block2
    each hold 4 (MHSA, MLP) pairs; Trans block3 holds 3 pairs. All three
    blocks operate on the same 257 x 768 token grid (256 patches + 1 class
    token), as stated in the paper ("The output sizes of all features from
    these three blocks are 257 x 768.").
    """

    def __init__(self, num_pairs, d_model=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.pairs = nn.ModuleList([
            MHSA_MLP_Pair(d_model, num_heads, mlp_ratio, dropout) for _ in range(num_pairs)
        ])

    def forward(self, x):
        for pair in self.pairs:
            x = pair(x)
        return x


# ---------------------------------------------------------------------------
# Feature Coupling Layers (Sec. III-B.3, Fig. 6)
# ---------------------------------------------------------------------------

class FeatureCouplingDown(nn.Module):
    """
    Feature coupling *down-sampling* layer: CNN branch -> Transformer
    branch (Fig. 6(a)).

    "The feature maps from the CNN branch are first put into a 1x1
     convolutional layer to align the channel numbers of the patch
     embeddings. Then down-sampling is implemented to complete the
     spatial dimension alignment through an average pooling layer with
     stride 4. The features are regularized by LayerNorm and GeLU layer."
    """

    def __init__(self, in_channels, d_model=768, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.channel_align = nn.Conv2d(in_channels, d_model, kernel_size=1)
        # AdaptiveAvgPool2d generalizes the paper's "stride-4 average
        # pooling" to any input resolution while always landing exactly
        # on the transformer's grid_size x grid_size token grid.
        self.pool = nn.AdaptiveAvgPool2d(grid_size)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, cnn_feat, tokens):
        """
        cnn_feat: (B, C, H, W)
        tokens:   (B, 1 + grid*grid, d_model) -- index 0 is the class token
        """
        x = self.channel_align(cnn_feat)
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, H*W, d_model)
        x = self.act(self.norm(x))

        cls_token, patch_tokens = tokens[:, :1], tokens[:, 1:]
        patch_tokens = patch_tokens + x
        return torch.cat([cls_token, patch_tokens], dim=1)


class FeatureCouplingUp(nn.Module):
    """
    Feature coupling *up-sampling* layer: Transformer branch -> CNN branch
    (Fig. 6(b)).

    "The patch embeddings are first arranged by the localization
     information of the patch to align the spatial scale S x S x E. Then
     the channel dimension is aligned with that of CNN feature maps
     through a 1x1 convolutional layer. Finally, up-sampling is performed
     on these features to align with those from the CNN branch. Meanwhile,
     batch normalization and LayerNorm are used to regularize features."
    """

    def __init__(self, out_channels, d_model=768, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.channel_align = nn.Conv2d(d_model, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, tokens, target_hw):
        """
        tokens:    (B, 1 + grid*grid, d_model)
        target_hw: (H, W) spatial size of the CNN feature map to be updated
        """
        patch_tokens = tokens[:, 1:]                    # drop the class token
        B, N, C = patch_tokens.shape
        S = self.grid_size
        x = patch_tokens.transpose(1, 2).contiguous().view(B, C, S, S)
        x = self.channel_align(x)
        x = self.bn(x)
        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x


# ---------------------------------------------------------------------------
# Pearson Correlation Layer (Sec. III-B.2, Eqs. (7)-(12))
# ---------------------------------------------------------------------------

class PearsonCorrelationLayer(nn.Module):
    """
    Computes, for a CNN feature map regarded as a grid of patch-like
    features (Eq. 7), the Pearson correlation coefficient between every
    pair of patches (Eqs. 8-10), sorts the correlation scores for each
    patch in descending order (Eq. 11) and keeps the top-K scores
    (Eq. 12), which are used downstream to localize potential copy-move
    regions.
    """

    def __init__(self, top_k=32):
        super().__init__()
        self.top_k = top_k

    def forward(self, feat):
        """
        feat: (B, C, H, W)
        returns: (B, H*W, top_k) pooled correlation scores per patch
        """
        B, C, H, W = feat.shape
        x = feat.flatten(2)                       # (B, C, N), N = H*W

        # Eq. (8): normalize each patch feature across the channel dim.
        mu = x.mean(dim=1, keepdim=True)           # (B, 1, N)
        sigma = x.std(dim=1, keepdim=True) + 1e-6
        x_norm = (x - mu) / sigma                  # (B, C, N)

        # Eq. (9)-(10): p(i, j) = f~[i]^T f~[j] / C for every pair (i, j)
        corr = torch.bmm(x_norm.transpose(1, 2), x_norm) / C   # (B, N, N)

        # Eq. (11)-(12): sort descending and keep the top-K scores per patch
        top_k = min(self.top_k, corr.shape[-1])
        pooled, _ = torch.topk(corr, k=top_k, dim=-1)
        return pooled                               # (B, N, top_k)
