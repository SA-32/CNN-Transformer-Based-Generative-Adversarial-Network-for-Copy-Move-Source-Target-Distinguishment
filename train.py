"""
Example training script for CNN-T GAN (Sec. IV-A: Experimental Setup).

Training hyperparameters follow the paper:
  * Adam optimizer (Kingma & Ba)
  * generator LR   = 1e-4
  * discriminator LR = 1e-5
  * batch size = 24
  * 100 epochs
  * per iteration: 1 discriminator update, then 2 generator updates

Replace `DummyCopyMoveDataset` with a real loader for USCISI / CASIA2 /
CoMoFoD (or your own copy-move dataset). Each sample must provide:
  image        : (3, 256, 256) copy-move forged image, normalized to [-1, 1]
  rgb_mask     : (3, 256, 256) ground-truth three-class (source/target/
                 pristine) mask, normalized to [-1, 1], channel order (B, G, R)
  binary_mask  : (1, 256, 256) ground-truth binary copy-move mask, in [0, 1]
"""
import torch
from torch.utils.data import Dataset, DataLoader

from cnn_t_gan.model import CNNTGAN


class DummyCopyMoveDataset(Dataset):
    """Placeholder dataset producing correctly-shaped random tensors, only
    meant to demonstrate the training loop end-to-end."""

    def __init__(self, length=100, img_size=256):
        self.length = length
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.rand(3, self.img_size, self.img_size) * 2 - 1
        rgb_mask = torch.rand(3, self.img_size, self.img_size) * 2 - 1
        binary_mask = torch.randint(0, 2, (1, self.img_size, self.img_size)).float()
        return img, rgb_mask, binary_mask


def train(num_epochs=100, batch_size=24, device=None, log_every=10):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNTGAN().to(device)

    opt_g = torch.optim.Adam(model.generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-5, betas=(0.9, 0.999))

    dataset = DummyCopyMoveDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(num_epochs):
        for step, (image, rgb_mask, binary_mask) in enumerate(loader):
            image = image.to(device)
            rgb_mask = rgb_mask.to(device)
            binary_mask = binary_mask.to(device)

            # ---- discriminator update (once per iteration) ----
            opt_d.zero_grad()
            d_loss = model.discriminator_step(image, rgb_mask)
            d_loss.backward()
            opt_d.step()

            # ---- generator update (twice per iteration) ----
            log = None
            for _ in range(2):
                opt_g.zero_grad()
                g_loss, log, _ = model.generator_step(image, rgb_mask, binary_mask)
                g_loss.backward()
                opt_g.step()

            if step % log_every == 0:
                print(
                    f"epoch {epoch:3d} step {step:4d} | "
                    f"D {d_loss.item():.4f} | "
                    f"G {log['total']:.4f} "
                    f"(adv {log['adv']:.4f}, mse {log['mse']:.4f}, "
                    f"mask {log['mask']:.4f}, simi {log['simi']:.4f})"
                )

    return model


if __name__ == '__main__':
    train(num_epochs=1, batch_size=2)   # small smoke-test defaults
