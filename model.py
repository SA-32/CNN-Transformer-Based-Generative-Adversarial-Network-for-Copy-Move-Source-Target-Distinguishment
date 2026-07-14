"""
Top-level CNN-T GAN model, wiring together the Generator, Discriminator
and losses defined in this package (Sec. III overall, Fig. 2).
"""
import torch
import torch.nn as nn

from .generator import Generator
from .discriminator import Discriminator
from .losses import CNNTGANLoss


class CNNTGAN(nn.Module):
    """
    Convenience wrapper implementing the alternating generator /
    discriminator training scheme described in Sec. IV-A:

        "the discriminator and the generator are trained alternatively.
         In an iteration, the discriminator is updated once and then the
         generator is updated twice."
    """

    def __init__(self, img_size=256, d_model=768, num_heads=12, grid_size=16,
                 pcl_top_k=32, mlp_ratio=4.0, dropout=0.1,
                 lambda1=100.0, lambda2=50.0, lambda3=20.0):
        super().__init__()
        self.generator = Generator(top_k = pcl_top_k, mlp_ratio = mlp_ratio)
        self.discriminator = Discriminator(in_channels=3 + 3)
        self.criterion = CNNTGANLoss(lambda1, lambda2, lambda3)

    def forward(self, x):
        return self.generator(x)

    def discriminator_step(self, image, gt_rgb_mask):
        """One discriminator update, Eq. (14)."""
        with torch.no_grad():
            gen_out = self.generator(image)
        d_real = self.discriminator(image, gt_rgb_mask)
        d_fake = self.discriminator(image, gen_out['rgb_mask'])
        loss = self.criterion.discriminator_loss(d_real, d_fake)
        return loss

    def generator_step(self, image, gt_rgb_mask, gt_binary_mask):
        """One generator update, Eq. (13)."""
        gen_out = self.generator(image)
        d_fake = self.discriminator(image, gen_out['rgb_mask'])
        loss, log = self.criterion.generator_loss(
            d_fake, gen_out['rgb_mask'], gt_rgb_mask,
            gen_out['binary_mask'], gt_binary_mask,
        )
        return loss, log, gen_out
