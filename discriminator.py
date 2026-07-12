"""
Discriminator of CNN-T GAN, Sec. III-C and Fig. 7.
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    "The structure of the discriminator is designed based on the
    discriminator of Patch-GAN... the discriminator consists of 4
    convolutional blocks and a fully connected layer. Each convolutional
    layer is followed by a BN layer and a LeakyReLU layer. The kernel
    sizes of all former 4 convolutional blocks are 5x5. Except that the
    last convolutional layer is of stride of 1, the former 3 convolutional
    layers are of a stride of 2. The output channels of the 4
    convolutional layers are 64, 128, 256, and 512, respectively. The
    output channel of the linear layer is 1. Finally, a sigmoid activation
    function and BCE loss are followed to predict whether the image pair
    is real or fake." (Sec. III-C)

    Input: concatenated (copy-move image, mask) pair, as in Fig. 2 /
    Fig. 7 ("image pairs" -> concatenate along channels).
    """

    def __init__(self, in_channels=6):
        super().__init__()

        def block(in_c, out_c, stride, use_bn=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=5, stride=stride, padding=2)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # First block has no BN, following the common PatchGAN convention
        # (the paper does not explicitly exempt it, but BN on raw
        # concatenated image/mask inputs is typically avoided).
        self.conv1 = block(in_channels, 64, stride=2, use_bn=False)
        self.conv2 = block(64, 128, stride=2)
        self.conv3 = block(128, 256, stride=2)
        self.conv4 = block(256, 512, stride=1)

        # The paper specifies a single fully-connected layer with output
        # channel 1; we global-average-pool the last feature map before
        # the linear layer so the discriminator works for any input size.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x).flatten(1)
        logit = self.fc(x)
        return torch.sigmoid(logit)         # probability that the pair is real
