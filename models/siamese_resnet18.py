"""Siamese U-Net with a pretrained ResNet-18 encoder.

The encoder is torchvision's ResNet-18, shared between the two dates, with the
first convolution widened from three channels to the band count of the product.
ImageNet weights are loaded for the three colour channels and the extra band is
initialised as their mean, so a four-band input starts from a sensible filter
rather than noise.

Fusion is the absolute difference of the two feature stacks at each of the four
encoder stages, the same idea as the baseline model. The decoder mirrors the
baseline: one upsampling step per stage, concatenating the difference skip at
that resolution.

This exists to answer one question: does a pretrained encoder help on 11 training
regions, or does the small from-scratch U-Net already capture what is there.

    python models/siamese_resnet18.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _decoder_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SiameseResNet18(nn.Module):
    """Shared ResNet-18 encoder, absolute-difference fusion, U-Net decoder.

    Parameters
    ----------
    in_channels : int
        Spectral bands per date.
    pretrained : bool
        Load ImageNet weights. Set False for tests and offline use.
    """

    def __init__(self, in_channels=4, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Widen the stem to the band count. With pretrained weights, keep the
        # colour channels and give the extra channels their mean.
        stem = backbone.conv1
        new_stem = nn.Conv2d(in_channels, stem.out_channels, kernel_size=7,
                             stride=2, padding=3, bias=False)
        with torch.no_grad():
            if in_channels >= 3:
                new_stem.weight[:, :3] = stem.weight
                if in_channels > 3:
                    extra = stem.weight.mean(dim=1, keepdim=True)
                    new_stem.weight[:, 3:] = extra.repeat(1, in_channels - 3, 1, 1)
            else:
                new_stem.weight[:, :in_channels] = stem.weight.mean(dim=1, keepdim=True)
        backbone.conv1 = new_stem

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                  backbone.maxpool)
        self.layer1 = backbone.layer1          # 64 channels,  1/4
        self.layer2 = backbone.layer2          # 128 channels, 1/8
        self.layer3 = backbone.layer3          # 256 channels, 1/16
        self.layer4 = backbone.layer4          # 512 channels, 1/32

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _decoder_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _decoder_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _decoder_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # No skip at 1/2, so dec1 takes the upsampled 32 channels directly.
        self.dec1 = _decoder_block(32, 32)
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final = _decoder_block(16, 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def _encode(self, x):
        x = self.stem(x)        # 1/4
        f1 = self.layer1(x)     # 1/4
        f2 = self.layer2(f1)    # 1/8
        f3 = self.layer3(f2)    # 1/16
        f4 = self.layer4(f3)    # 1/32
        return f1, f2, f3, f4

    def forward(self, x1, x2):
        a1, a2, a3, a4 = self._encode(x1)
        b1, b2, b3, b4 = self._encode(x2)
        d1 = torch.abs(a1 - b1)
        d2 = torch.abs(a2 - b2)
        d3 = torch.abs(a3 - b3)
        d4 = torch.abs(a4 - b4)

        x = self.dec4(torch.cat([self.up4(d4), d3], dim=1))    # 1/16
        x = self.dec3(torch.cat([self.up3(x), d2], dim=1))     # 1/8
        x = self.dec2(torch.cat([self.up2(x), d1], dim=1))     # 1/4
        x = self.dec1(self.up1(x))                             # 1/2
        x = self.final(self.final_up(x))                       # 1/1
        return self.head(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = SiameseResNet18(in_channels=4, pretrained=False)
    x1 = torch.randn(2, 4, 128, 128)
    x2 = torch.randn(2, 4, 128, 128)
    with torch.no_grad():
        out = model(x1, x2)
    print("output shape:", tuple(out.shape))
    print("parameters: %.2f M" % (model.parameter_count() / 1e6))
    print("fp32 weights: %.1f MB" % (model.parameter_count() * 4 / 1e6))
