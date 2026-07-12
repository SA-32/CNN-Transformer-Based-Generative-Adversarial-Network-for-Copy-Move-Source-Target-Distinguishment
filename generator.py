"""
Generator of CNN-T GAN, Sec. III-B and Figs. 3-4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    ConvStage,
    TransformerStage,
    FeatureCouplingDown,
    FeatureCouplingUp,
    PearsonCorrelationLayer,
)


class PreprocessingBlock(nn.Module):
    """
    Sec. III-B: "The pre-processing block first reshapes the images into
    the size of 256x256x3, and then filters the images with a
    convolutional layer with kernel size of 7x7 and stride of 2, followed
    by a batch normalization (BN) layer and a rectified linear units
    (ReLU) layer. Finally, a max pooling layer with stride 2 is used to
    halve the size of the output feature. The size of the output feature
    of the pre-processing block is 64x64x64."
    """

    def __init__(self, in_channels=3, out_channels=64, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                               mode='bilinear', align_corners=False)
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x                       # (B, 64, 64, 64)


class PatchEmbedding(nn.Module):
    """
    "the features output from the pre-processing block are first put into
    a convolutional layer with kernel size of 4x4 and stride 4. Then four
    pairs of multi-heads self-attention (MHSA) blocks and multilayer
    perceptron (MLP) layers are followed..." (Sec. III-B.1)

    64x64 features with a stride-4, 4x4 conv give a 16x16 = 256 patch
    grid; a learnable class token is prepended, giving the 257-token
    sequence used throughout the transformer branch.
    """

    def __init__(self, in_channels=64, d_model=768, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=4, stride=4)
        num_patches = grid_size * grid_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x)                         # (B, d_model, 16, 16)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, 256, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)            # (B, 257, d_model)
        x = x + self.pos_embed
        return x


class Generator(nn.Module):
    """
    Generator of CNN-T GAN (Sec. III-B, Figs. 3 & 4).

    The generator is composed of:
      * a pre-processing block
      * a CNN branch (local features) with 3 conv stages: 256x64x64,
        512x32x32, 1024x16x16
      * a transformer branch (global features) with 3 blocks of
        (4, 4, 3) (MHSA, MLP) pairs operating on a fixed 257x768 token grid
      * feature coupling layers (down: CNN->Transformer, up:
        Transformer->CNN) attached at every stage
      * two Pearson correlation layers (on Conv block2 and Conv block3
        features) used to build the binary copy-move similarity mask

    It produces two outputs, as described in Sec. III-A ("the generator
    is utilized to generate a binary mask and an RGB mask"):
      * rgb_mask    : (B, 3, H, W), tanh-activated three-class
                      source/target/pristine map, trained with
                      L_MSE + L_mask (Eqs. 15-16). Channel order is
                      assumed (B, G, R) as in the paper, where only the
                      G (source) and R (target) channels carry the
                      re-weighted mask loss.
      * binary_mask : (B, 1, H, W), sigmoid-activated copy-move
                      similarity map built from the Pearson-correlation
                      features, trained with L_simi (Eq. 17).
    """

    def __init__(self, img_size=256, d_model=768, num_heads=12, grid_size=16,
                 pcl_top_k=32, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.preprocess = PreprocessingBlock(3, 64, img_size)
        self.patch_embed = PatchEmbedding(64, d_model, grid_size)

        # CNN branch: 3 conv stages -> 256x64x64, 512x32x32, 1024x16x16
        self.conv_stage1 = ConvStage(64, 256, num_bottlenecks=7, downsample=False)
        self.conv_stage2 = ConvStage(256, 512, num_bottlenecks=8, downsample=True)
        self.conv_stage3 = ConvStage(512, 1024, num_bottlenecks=6, downsample=True)

        # Transformer branch: 3 blocks with 4, 4, 3 (MHSA, MLP) pairs
        self.trans_stage1 = TransformerStage(4, d_model, num_heads, mlp_ratio, dropout)
        self.trans_stage2 = TransformerStage(4, d_model, num_heads, mlp_ratio, dropout)
        self.trans_stage3 = TransformerStage(3, d_model, num_heads, mlp_ratio, dropout)

        # Feature coupling layers, one down/up pair per stage (Fig. 6)
        self.fcl_down1 = FeatureCouplingDown(256, d_model, grid_size)
        self.fcl_up1 = FeatureCouplingUp(256, d_model, grid_size)
        self.fcl_down2 = FeatureCouplingDown(512, d_model, grid_size)
        self.fcl_up2 = FeatureCouplingUp(512, d_model, grid_size)
        self.fcl_down3 = FeatureCouplingDown(1024, d_model, grid_size)
        self.fcl_up3 = FeatureCouplingUp(1024, d_model, grid_size)

        # Pearson correlation layers on Conv block2 / Conv block3 features
        self.pcl2 = PearsonCorrelationLayer(top_k=pcl_top_k)
        self.pcl3 = PearsonCorrelationLayer(top_k=pcl_top_k)

        # Head that turns the fused multi-scale CNN features into the
        # RGB three-class mask ("the fused feature maps are put into a
        # tanh activation function").
        self.rgb_head = nn.Sequential(
            nn.Conv2d(256 + 512 + 1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=1),
            nn.Tanh(),
        )

        # Heads that turn the concatenated PCL outputs into the binary
        # copy-move similarity mask ("a binary cross-entropy loss is
        # utilized to acquire the similar regions in images").
        self.simi_head2 = nn.Linear(pcl_top_k, 1)
        self.simi_head3 = nn.Linear(pcl_top_k, 1)
        self.simi_fuse = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, x):
        target_size = x.shape[-2:]

        feat0 = self.preprocess(x)                       # (B, 64, 64, 64)
        tokens = self.patch_embed(feat0)                  # (B, 257, 768)
        cnn_feat = feat0

        # ---- stage 1 (Conv block1 <-> Trans block1) ----
        cnn_feat = self.conv_stage1(cnn_feat)             # (B, 256, 64, 64)
        tokens = self.trans_stage1(tokens)
        tokens = self.fcl_down1(cnn_feat, tokens)
        cnn_feat = cnn_feat + self.fcl_up1(tokens, cnn_feat.shape[-2:])
        feat_c1 = cnn_feat

        # ---- stage 2 (Conv block2 <-> Trans block2) ----
        cnn_feat = self.conv_stage2(cnn_feat)             # (B, 512, 32, 32)
        tokens = self.trans_stage2(tokens)
        tokens = self.fcl_down2(cnn_feat, tokens)
        cnn_feat = cnn_feat + self.fcl_up2(tokens, cnn_feat.shape[-2:])
        feat_c2 = cnn_feat

        # ---- stage 3 (Conv block3 <-> Trans block3) ----
        cnn_feat = self.conv_stage3(cnn_feat)             # (B, 1024, 16, 16)
        tokens = self.trans_stage3(tokens)
        tokens = self.fcl_down3(cnn_feat, tokens)
        cnn_feat = cnn_feat + self.fcl_up3(tokens, cnn_feat.shape[-2:])
        feat_c3 = cnn_feat

        # ---- multi-scale fusion -> RGB (three-class) mask ----
        up1 = F.interpolate(feat_c1, size=target_size, mode='bilinear', align_corners=False)
        up2 = F.interpolate(feat_c2, size=target_size, mode='bilinear', align_corners=False)
        up3 = F.interpolate(feat_c3, size=target_size, mode='bilinear', align_corners=False)
        fused = torch.cat([up1, up2, up3], dim=1)
        rgb_mask = self.rgb_head(fused)

        # ---- Pearson-correlation branch -> binary similarity mask ----
        pcl_out2 = self.pcl2(feat_c2)                     # (B, N2, K)
        pcl_out3 = self.pcl3(feat_c3)                     # (B, N3, K)

        s2 = self.simi_head2(pcl_out2)                    # (B, N2, 1)
        s3 = self.simi_head3(pcl_out3)                    # (B, N3, 1)

        B = x.shape[0]
        H2, W2 = feat_c2.shape[-2:]
        H3, W3 = feat_c3.shape[-2:]
        s2 = s2.transpose(1, 2).contiguous().view(B, 1, H2, W2)
        s3 = s3.transpose(1, 2).contiguous().view(B, 1, H3, W3)
        s2 = F.interpolate(s2, size=target_size, mode='bilinear', align_corners=False)
        s3 = F.interpolate(s3, size=target_size, mode='bilinear', align_corners=False)

        simi_logits = self.simi_fuse(torch.cat([s2, s3], dim=1))
        binary_mask = torch.sigmoid(simi_logits)

        return {
            'rgb_mask': rgb_mask,          # (B, 3, H, W), tanh in [-1, 1]
            'binary_mask': binary_mask,    # (B, 1, H, W), sigmoid in [0, 1]
        }
