"""
Loss functions of CNN-T GAN, Sec. III-D, Eqs. (13)-(17).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    
    def forward(self, d_real, d_fake):
        eps = 1e-8
        real_loss = -torch.log(d_real.clamp(min=eps)).mean()
        fake_loss = -torch.log((1 - d_fake).clamp(min=eps)).mean()
        return real_loss + fake_loss

    def generator_loss(self, d_fake):
        # generator wants D(X, G(X)) -> 1, i.e. maximize log(D(X, G(X)))
        eps = 1e-8
        return -torch.log(d_fake.clamp(min=eps)).mean()


class MSELoss(nn.Module):
    
    def forward(self, gen, gt):
        return F.mse_loss(gen, gt)


class MaskLoss(nn.Module):
    """
    Eq. (16): L_mask = L_MSE (elementwise) * mask, with
    mask = [0, 1, 1] over the (B, G, R) channels of the RGB mask, so that
    only the source (G) and target (R) channels contribute
    ("only the channels G and R are taken into consideration").
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('channel_weight', torch.tensor([0.0, 1.0, 1.0]).view(1, 3, 1, 1))

    def forward(self, gen, gt):
        diff2 = (gen - gt) ** 2
        weighted = diff2 * self.channel_weight.to(gen.device)
        return weighted.mean()


class SimilarityLoss(nn.Module):
    
    def forward(self, pred_binary, gt_binary):
        pred_binary = pred_binary.clamp(1e-6, 1 - 1e-6)
        return F.binary_cross_entropy(pred_binary, gt_binary)


class CNNTGANLoss(nn.Module):
    
    def __init__(self, lambda1=100.0, lambda2=50.0, lambda3=20.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.adv  = AdversarialLoss()
        self.mse  = MSELoss()
        self.mask = MaskLoss()
        self.simi = SimilarityLoss()

    def discriminator_loss(self, d_real, d_fake):
        return self.adv(d_real, d_fake)

    def generator_loss(self, d_fake, gen_rgb, gt_rgb, gen_binary, gt_binary):
        l_adv  = self.adv.generator_loss(d_fake)
        l_mse  = self.mse(gen_rgb, gt_rgb)
        l_mask = self.mask(gen_rgb, gt_rgb)
        l_simi = self.simi(gen_binary, gt_binary)

        total = (l_adv
                 + self.lambda1 * l_mse
                 + self.lambda2 * l_mask
                 + self.lambda3 * l_simi)

        logs = {
            'adv'  : l_adv.item(),
            'mse'  : l_mse.item(),
            'mask' : l_mask.item(),
            'simi' : l_simi.item(),
            'total': total.item(),
        }
        return total, logs
