import torch


class Mul(torch.nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.module(x)


def _conv_bn(
    channels_in: int,
    channels_out: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(inplace=True),
    )


def resnet9(num_classes: int = 10, channels_last: bool = True) -> torch.nn.Module:
    """ResNet9 architecture used for CIFAR-10 in EasyDUB."""
    model: torch.nn.Module = torch.nn.Sequential(
        _conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        _conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(_conv_bn(128, 128), _conv_bn(128, 128))),
        _conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(_conv_bn(256, 256), _conv_bn(256, 256))),
        _conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2),
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model

