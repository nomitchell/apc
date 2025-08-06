# /apc_project/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class PurifierUNet(nn.Module):
    """A 3-Level U-Net for 32x32 image purification."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1_1 = ConvBlock(3, 64)
        self.enc1_2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = ConvBlock(64, 128)
        self.enc2_2 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = ConvBlock(128, 256)
        self.enc3_2 = ConvBlock(256, 256)

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_1 = ConvBlock(256, 128) # 128 from upconv2 + 128 from enc2
        self.dec2_2 = ConvBlock(128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_1 = ConvBlock(128, 64) # 64 from upconv1 + 64 from enc1
        self.dec1_2 = ConvBlock(64, 64)

        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1_2(self.enc1_1(x))
        e2 = self.enc2_2(self.enc2_1(self.pool1(e1)))
        e3 = self.enc3_2(self.enc3_1(self.pool2(e2)))

        # Decoder
        d2 = self.upconv2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_2(self.dec2_1(d2))

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_2(self.dec1_1(d1))

        out = self.out_conv(d1)
        return out

class BasicBlock(nn.Module):
    """Basic Block for WideResNet."""
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes,
                                                                kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet34_10(nn.Module):
    """WRN-34-10 model."""
    def __init__(self, num_classes=10, dropRate=0.3):
        super(WideResNet34_10, self).__init__()
        n = 5 # (34 - 4) / 6 = 5 blocks per layer
        k = 10 # Widen factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlock, 1, dropRate)
        self.layer2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlock, 2, dropRate)
        self.layer3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlock, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)
        self.nChannels = nStages[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class ComposedModel(nn.Module):
    """Wrapper to chain the Purifier and Classifier for the attacker."""
    def __init__(self, purifier, classifier):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.purifier(x))