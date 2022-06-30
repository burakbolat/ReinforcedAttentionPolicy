import torch
import torch.nn as nn
from torch import Tensor

class ConvBlock(nn.Module):
    '''
    Conv Block Architecture used in 
    Matching Networks for One Shot Learning and
    Prototypical Networks for Few-shot Learning
    '''
    def __init__(self,
                 inplanes: int,
                 planes: int = 64) -> None:
        super().__init__()
        ## in Matching Nets paper the authors says that
        ## when 4 of these blocks used with and 28x28 rgb image
        ## the output is 1x1 with 64 for channels.
        ## for the output size to match a padding of 1 is used
        ## which is not discussed in the paper.
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class Conv4(nn.Module):
    def __init__(self, inplanes: int = 3) -> None:
        super().__init__()
        self.cb1 = ConvBlock(inplanes, 64)
        self.cb2 = ConvBlock(64, 64)
        self.cb3 = ConvBlock(64, 64)
        self.cb4 = ConvBlock(64, 64)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        return x

class Conv6(nn.Module):
    def __init__(self, inplanes: int = 3) -> None:
        super().__init__()
        self.cb1 = ConvBlock(inplanes, 64)
        self.cb2 = ConvBlock(64, 64)
        self.cb3 = ConvBlock(64, 64)
        self.cb4 = ConvBlock(64, 64)
        self.cb5 = ConvBlock(64, 64)
        self.cb6 = ConvBlock(64, 64)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        x = self.cb5(x)
        x = self.cb6(x)
        return x


if __name__ == "__main__":
    conv4 = Conv4(3)
    image = torch.rand((1,3,84,84))
    feature = conv4(image)
    print(feature.size())
