import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model = 768, num_heads = 12, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x



# ==================== 2. Transformer ====================
class TransformerBlock(nn.Module):
    def __init__(self, d_model = 768, num_heads = 12, mlp_ratio = 2.66, dropout = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FeatureCouplingDown(nn.Module):
    def __init__(self, cnn_channels, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(cnn_channels, embed_dim, 1)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, H, W] from CNN
        x = self.conv(x)      # [B, E, H, W]
        x = self.pool(x)      # [B, E, H/4, W/4]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        x = self.norm(x)
        x = self.act(x)
        return x

class FeatureCouplingUp(nn.Module):
    def __init__(self, embed_dim, cnn_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, cnn_channels, 1)
        self.upsample = nn.Upsample(scale_factor = scale_factor, mode='bilinear', align_corners=False)
        self.norm = nn.BatchNorm2d(cnn_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, N, E] from Transformer (excluding cls token)
        B, N, E = x.shape

        x = x.transpose(1, 2).reshape(B, E, int(N**0.5), int(N**0.5))
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.upsample(x)
        return x

class Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4, stride = 2, padding = 1)
        )

    def forward(self, x):
        return self.model(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, padding = 0, stride = 1),
            nn.BatchNorm2d(64, eps = 1e-06),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64, eps = 1e-06),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size = 1, padding = 0, stride = 1),
            nn.BatchNorm2d(256, eps = 1e-06),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class Block1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 4, padding = 1, stride = 2),
            nn.BatchNorm2d(output_channels, eps = 1e-06),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class ConvBlock_1_part_1(nn.Module):
    def __init__(self, input_channels, hidden_channels, scale_factor, m_ratio):
        super().__init__()
        self.cv_block_1_c1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = hidden_channels, kernel_size = 1, padding = 0, stride = 1),
            nn.BatchNorm2d(hidden_channels, eps = 1e-06),
            nn.ReLU(),

            nn.Conv2d(in_channels = hidden_channels, out_channels = hidden_channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(hidden_channels, eps = 1e-06),
            nn.ReLU()
        )

        self.cv_block_1_c2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_channels, out_channels = input_channels, kernel_size = 1, padding = 0, stride = 1),
            nn.BatchNorm2d(input_channels, eps = 1e-06),
            nn.ReLU()
        )

        self.cv_block_1_c3 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = hidden_channels, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(hidden_channels, eps = 1e-06),
            nn.ReLU()
        )

        self.cv_block_1_c4 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_channels, out_channels = hidden_channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(hidden_channels, eps = 1e-06),
            nn.ReLU(),

            nn.Conv2d(in_channels = hidden_channels, out_channels = input_channels, kernel_size = 1, padding = 0, stride = 1),
            nn.BatchNorm2d(input_channels, eps = 1e-06),
            nn.ReLU()
        )

        self.transformer_block_1_part_1 = TransformerBlock(mlp_ratio = m_ratio)

        self.transformer_block_1_part_2 = TransformerBlock(mlp_ratio = m_ratio)

        self.fcl_down_1 = FeatureCouplingDown(hidden_channels, 768)

        self.fcl_up_1 = FeatureCouplingUp(768, hidden_channels, scale_factor)


    def forward(self, y, x):
        z = self.transformer_block_1_part_1(x)
        out_2 = self.cv_block_1_c1(y)
        z = z + self.fcl_down_1(out_2)
        z = self.transformer_block_1_part_2(z)
        out_3 = self.cv_block_1_c2(out_2)
        out_3 = y + out_3
        out_4 = self.cv_block_1_c3(out_3)
        out_4 = out_4 + self.fcl_up_1(z)
        out_5 = self.cv_block_1_c4(out_4)
        out_0 = out_5 + out_3

        return out_0, z

class ConvBlock_1(nn.Module):
    def __init__(self, m_ratio):
        super().__init__()
        self.trans_1_conv = nn.Conv2d(in_channels = 64, out_channels = 768, kernel_size = 4, stride = 4)
        self.model1 = ConvBlock_1_part_1(input_channels = 256, hidden_channels = 64, scale_factor = 4, m_ratio = m_ratio)
        self.model2 = ConvBlock_1_part_1(input_channels = 256, hidden_channels = 64, scale_factor = 4, m_ratio = m_ratio)
        self.model3 = ConvBlock_1_part_1(input_channels = 256, hidden_channels = 64, scale_factor = 4, m_ratio = m_ratio)


    def forward(self, y, x):
        x = self.trans_1_conv(x).flatten(2).permute(0,2,1)
        y, x = self.model1(y, x)
        y, x = self.model2(y, x)
        y, x = self.model3(y, x)
        return y, x

class ConvBlock_2(nn.Module):
    def __init__(self, m_ratio):
        super().__init__()
        self.model1 = ConvBlock_1_part_1(input_channels = 512, hidden_channels = 128, scale_factor = 2, m_ratio = m_ratio)
        self.model2 = ConvBlock_1_part_1(input_channels = 512, hidden_channels = 128, scale_factor = 2, m_ratio = m_ratio)
        self.model3 = ConvBlock_1_part_1(input_channels = 512, hidden_channels = 128, scale_factor = 2, m_ratio = m_ratio)
        self.model4 = ConvBlock_1_part_1(input_channels = 512, hidden_channels = 128, scale_factor = 2, m_ratio = m_ratio)

    def forward(self, y, x):
        y, x = self.model1(y, x)
        y, x = self.model2(y, x)
        y, x = self.model3(y, x)
        y, x = self.model4(y, x)
        return y, x

class ConvBlock_3(nn.Module):
    def __init__(self, m_ratio):
        super().__init__()
        self.model1 = ConvBlock_1_part_1(input_channels = 1024, hidden_channels = 256, scale_factor = 1, m_ratio = m_ratio)
        self.model2 = ConvBlock_1_part_1(input_channels = 1024, hidden_channels = 256, scale_factor = 1, m_ratio = m_ratio)
        self.model3 = ConvBlock_1_part_1(input_channels = 1024, hidden_channels = 256, scale_factor = 1, m_ratio = m_ratio)

    def forward(self, y, x):
        y, x = self.model1(y, x)
        y, x = self.model2(y, x)
        y, x = self.model3(y, x)
        return y, x

class PearsonCorrelationLayer(nn.Module):
    """
    Computes patch-wise self-correlation of a CxHxW feature map (Eq. 7),
    normalizes each patch vector to zero-mean/unit-std across channels
    (Eq. 8, Assumption A6), forms the Pearson correlation score matrix
    (Eq. 9-10), and returns the top-K correlation scores per patch,
    reshaped back to a (K, H, W) spatial map (Eq. 11-12).
    """

    def __init__(self, topk: int = 16):
        super().__init__()
        self.topk = topk

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        N = H * W
        f = feat.flatten(2)                       # (B, C, N)

        mu = f.mean(dim=1, keepdim=True)           # per-patch mean over channels
        sigma = f.std(dim=1, keepdim=True) + 1e-6  # per-patch std over channels
        f_norm = (f - mu) / sigma                  # Eq. (8)

        # Eq. (9)-(10): Pearson correlation between every pair of patches
        corr = torch.bmm(f_norm.transpose(1, 2), f_norm) / C   # (B, N, N)

        # Eq. (11): sort descending; Eq. (12): keep top-K (excluding self-match
        # at k=0 which is always 1.0 and uninformative)
        k = min(self.topk + 1, N)
        topk_scores, _ = torch.topk(corr, k=k, dim=-1)
        topk_scores = topk_scores[..., 1:]          # drop the trivial self-score

        pooled = topk_scores.transpose(1, 2).reshape(B, self.topk, H, W)
        return pooled

class Generator(nn.Module):
    def __init__(self, top_k = 16, mlp_ratio = 2.66):
        super().__init__()

        self.preprocess = Preprocessor()

        self.conv1 = Block()

        self.cv1   = ConvBlock_1(m_ratio = mlp_ratio)

        self.conv2 = Block1(256, 512)

        self.cv2   = ConvBlock_2(m_ratio = mlp_ratio)

        self.conv3 = Block1(512, 1024)

        self.cv3   = ConvBlock_3(m_ratio = mlp_ratio)

        fused_channels = 256 + 512 + 1024
        self.rgb_head = nn.Sequential(
            nn.Conv2d(fused_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.pcl2 = PearsonCorrelationLayer(top_k)
        self.pcl3 = PearsonCorrelationLayer(top_k)


        self.mask_head = nn.Sequential(
            nn.Conv2d(top_k * 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, _, _, _ = x.shape

        x = self.preprocess(x)

        y = self.conv1(x)

        y, x = self.cv1(y, x)

        conv1_out = y

        y = self.conv2(y)

        y, x = self.cv2(y, x)

        conv2_out = y

        y = self.conv3(y)

        y, x = self.cv3(y, x)

        conv3_out = y

        up1 = F.interpolate(conv1_out, size = 256, mode="bilinear", align_corners=False)
        up2 = F.interpolate(conv2_out, size = 256, mode="bilinear", align_corners=False)
        up3 = F.interpolate(conv3_out, size = 256, mode="bilinear", align_corners=False)
        fused = torch.cat([up1, up2, up3], dim=1)
        rgb_mask = self.rgb_head(fused)

        pcl2 = self.pcl2(conv2_out)
        pcl3 = self.pcl3(conv3_out)


        pcl2_up = F.interpolate(pcl2, size = 256, mode="bilinear", align_corners=False)
        pcl3_up = F.interpolate(pcl3, size = 256, mode="bilinear", align_corners=False)
        pcl_fused = torch.cat([pcl2_up, pcl3_up], dim=1)
        binary_mask = self.mask_head(pcl_fused)

        return {
            'rgb_mask': rgb_mask,          # (B, 3, H, W), tanh in [-1, 1]
            'binary_mask': binary_mask,    # (B, 1, H, W), sigmoid in [0, 1]
        }