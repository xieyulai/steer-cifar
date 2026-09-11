"""WideResNet-28-10 (Zagoruyko & Komodakis, BMVC 2016).

Pre-activation residual network with width factor k=10 and depth=28 (= 6n+4, n=4
basic blocks per stage). dropout=0.0 保持单变量清洁（vs R8 leader DLA 也 dropout=0）。
channels: [16, 160, 320, 640]；3 stages × 4 blocks；~36.5M params on CIFAR-10.

Reference: Zagoruyko & Komodakis, "Wide Residual Networks" (BMVC 2016).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasic(nn.Module):
    """Pre-activation basic block: BN→ReLU→Conv 3x3 → BN→ReLU→Dropout→Conv 3x3 + shortcut."""
    def __init__(self, in_planes: int, planes: int, dropout_rate: float, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """WideResNet-N-k: depth=N (N=6n+4), widening factor k."""
    def __init__(self, depth: int = 28, widen_factor: int = 10, dropout_rate: float = 0.0,
                 num_classes: int = 10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4 (e.g. 28, 40, 16)"
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.in_planes = n_stages[0]
        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasic, n_stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, n_stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, n_stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(n_stages[3])
        self.linear = nn.Linear(n_stages[3], num_classes)

    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def WRN28_10(num_classes: int = 10) -> WideResNet:
    """WRN-28-10: depth 28, widening factor 10, ~36.5M params on CIFAR-10."""
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=num_classes)
