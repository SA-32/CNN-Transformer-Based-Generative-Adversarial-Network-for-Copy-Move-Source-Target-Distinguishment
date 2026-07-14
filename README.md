# CNN-T GAN — PyTorch Implementation

PyTorch implementation of:

Y. Zhang, G. Zhu, X. Wang, X. Luo, Y. Zhou, H. Zhang, L. Wu,
**"CNN-Transformer Based Generative Adversarial Network for Copy-Move
Source/Target Distinguishment,"** *IEEE Transactions on Circuits and
Systems for Video Technology*, vol. 33, no. 5, pp. 2019–2031, May 2023.

This follows the architecture in Sec. III of the paper (Figs. 2–7) as
closely as the text allows. A few implementation details are **not fully
specified in the paper** and were filled in with standard, clearly
documented choices — see "Assumptions" below.

## Package layout

```
cnn_t_gan/
├── __init__.py
├── generator.py       # PreprocessingBlock, PatchEmbedding, Generator (Fig. 3-4)
├── discriminator.py   # PatchGAN-style Discriminator (Fig. 7)
├── losses.py          # AdversarialLoss, MSELoss, MaskLoss, SimilarityLoss,
│                      # CNNTGANLoss (Eqs. 13-17)
├── model.py           # CNNTGAN wrapper (generator + discriminator + losses)
└── train.py           # example training loop (Sec. IV-A hyperparameters)
```

## Quick start

```python
import torch
from cnn_t_gan.model import CNNTGAN

model = CNNTGAN().cuda()

image = torch.randn(4, 3, 256, 256).cuda()          # copy-move image, in [-1, 1]
gt_rgb_mask = torch.randn(4, 3, 256, 256).cuda()     # GT 3-class mask, in [-1, 1], (B,G,R)
gt_binary_mask = torch.randint(0, 2, (4, 1, 256, 256)).float().cuda()

# discriminator update
d_loss = model.discriminator_step(image, gt_rgb_mask)
d_loss.backward()

# generator update
g_loss, log, outputs = model.generator_step(image, gt_rgb_mask, gt_binary_mask)
g_loss.backward()

# inference
model.eval()
with torch.no_grad():
    out = model(image)
    rgb_mask = out['rgb_mask']        # (B, 3, 256, 256), tanh output
    binary_mask = out['binary_mask']  # (B, 1, 256, 256), sigmoid output
```

Run the example training loop (uses a dummy random dataset — swap in
USCISI / CASIA2 / CoMoFoD):

```bash
python -m cnn_t_gan.train
```

## Architecture summary (mapped to the paper)

| Component | Paper section | Shapes |
|---|---|---|
| Pre-processing block | Sec. III-B, first paragraph | 256×256×3 → 64×64×64 |
| Transformer branch | Sec. III-B.1, Fig. 4 top | patch conv 4×4/s4 → 16×16 grid → 257×768 tokens; 3 blocks of (4,4,3) MHSA+MLP pairs |
| CNN branch | Sec. III-B.2, Fig. 4 bottom | Conv block1 (7 bottlenecks) → 256×64×64; Conv block2 (8 bottlenecks) → 512×32×32; Conv block3 (6 bottlenecks) → 1024×16×16 |
| Feature Coupling Layers | Sec. III-B.3, Fig. 6 | one down (CNN→Trans) + one up (Trans→CNN) pair per stage |
| Pearson Correlation Layer | Sec. III-B.2, Eqs. 7-12 | applied to Conv block2 and Conv block3 features |
| Generator outputs | Sec. III-A | `rgb_mask` (tanh, 3-class source/target/pristine), `binary_mask` (sigmoid, copy-move similarity) |
| Discriminator | Sec. III-C, Fig. 7 | 4 conv blocks (5×5, strides 2/2/2/1, channels 64/128/256/512) + FC + sigmoid |
| Loss | Sec. III-D, Eqs. 13-17 | `L_total = L_adv + 100·L_MSE + 50·L_mask + 20·L_simi` |

## Assumptions / details not fully specified in the paper

The paper describes the architecture at a level that pins down most
shapes and hyperparameters, but leaves some implementation choices open.
These are called out in code comments; summarized here:

1. **Bottleneck reduction ratio.** The paper specifies 1×1 down-project →
   3×3 → 1×1 up-project bottlenecks but not the channel-reduction ratio.
   We use the standard ResNet ratio of 4 (e.g. 256-channel bottleneck ->
   64 mid-channels).
2. **Where CNN-branch downsampling happens.** The paper gives output
   sizes 256×64×64 → 512×32×32 → 1024×16×16 for the three conv blocks but
   not exactly which bottleneck performs the stride-2 downsampling.
3. **Transformer MLP expansion ratio.** Not stated; we use the standard
   ViT ratio of 4× (768 → 3072 → 768).
4. **Feature Coupling Layer exact pooling/upsampling factors.** The paper
   states "average pooling with stride 4" for the down-sampling FCL, which
   matches exactly for the 64×64 → 16×16 case (stage 1) but not for
   stages 2/3 (32×32 or 16×16 spatial input). We use `AdaptiveAvgPool2d`
   (down) / bilinear `F.interpolate` (up) so the same module generalizes
   correctly to every stage while matching the paper exactly where it is
   fully specified.
5. **PCL top-K value.** The paper defines the top-K pooling operation
   (Eq. 12) but does not give a numeric K; it is exposed as the
   `pcl_top_k` constructor argument (default 32).
6. **How the two PCL outputs become the final binary mask.** The paper
   says the two PCL outputs are "fused by concatenation, and finally a
   binary cross-entropy loss is utilized to acquire the similar regions."
   We implement this as: a linear layer maps each patch's top-K score
   vector to a single logit, the two resulting (differently-sized) maps
   are upsampled to the input resolution, concatenated, and fused with a
   1×1 conv + sigmoid to produce the final `binary_mask` used in `L_simi`.
7. **RGB mask channel order.** The paper explicitly states the mask-loss
   channel weights `[0, 1, 1]` correspond to `(B, G, R)`, i.e. only the
   green (source) and red (target) channels are weighted; we keep this
   convention in `MaskLoss` and it should be matched by your data
   pipeline's ground-truth mask channel order.
8. **Discriminator's final "linear layer."** For an input-size-agnostic
   implementation we global-average-pool the last conv feature map
   before the `Linear(512, 1)` layer (a `Linear` on a flattened,
   fixed-resolution feature map would also match the paper and is a
   one-line change if you prefer that).

## Compute note

This is a large network by design — the transformer branch alone uses
`d_model=768`, 12 attention heads and a 257-token sequence, run alongside
a CNN branch with up to 1024 channels, all at up to 256×256 spatial
resolution, matching the paper's stated configuration. The paper trained
with batch size 24 on an NVIDIA RTX 2080 Ti (11 GB); plan for a GPU with
comparable or greater memory, and consider mixed-precision training
(`torch.cuda.amp`) if you hit memory limits.
