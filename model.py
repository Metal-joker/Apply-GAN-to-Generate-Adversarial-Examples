import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, input_c, output_c):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_c, output_c, kernel_size=(3, 3), padding=1, bias=False),
            nn.InstanceNorm2d(output_c, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_c, output_c, kernel_size=(3, 3), padding=1, bias=False),
            nn.InstanceNorm2d(output_c, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.main(x)

class NormalBlock(nn.Module):
    def __init__(self, input_c, output_c):
        super(NormalBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_c, output_c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        return self.main(x)


class ResGenerator(nn.Module):
    def __init__(self, num_classes=5, num_filters=64, num_downsample=2, num_bottleneck=6):
        super(ResGenerator, self).__init__()

        layer = []
        layer.append(nn.Conv2d(3 + num_classes, num_filters, kernel_size=7, padding=3, bias=False))
        layer.append(nn.InstanceNorm2d(num_filters, affine=True, track_running_stats=True))
        layer.append(nn.ReLU(inplace=True))

        curr_num_filters = num_filters
        for i in range(num_downsample):
            layer.append(nn.Conv2d(curr_num_filters, curr_num_filters * 2, kernel_size=4, padding=1, bias=False))
            layer.append(nn.InstanceNorm2d(curr_num_filters * 2, affine=True, track_running_stats=True))
            layer.append(nn.ReLU(inplace=True))
            curr_num_filters = curr_num_filters * 2

        for i in range(num_bottleneck):
            layer.append(ResBlock(curr_num_filters, curr_num_filters))

        for i in range(num_downsample):
            layer.append(nn.ConvTranspose2d(curr_num_filters, curr_num_filters // 2, kernel_size=4, padding=1, bias=False))
            layer.append(nn.InstanceNorm2d(curr_num_filters // 2, affine=True, track_running_stats=True))
            layer.append(nn.ReLU(inplace=True))
            curr_num_filters = curr_num_filters // 2
        
        layer.append(nn.Conv2d(curr_num_filters, 3, kernel_size=7, padding=3, bias=False))
        layer.append(nn.Tanh())
        
        self.main = nn.Sequential(*layer)        

    def forward(self, x, target_label):
        target_label = target_label.view(target_label.size(0), target_label.size(1), 1, 1)
        target_label = target_label.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, target_label], dim=1)
        del target_label
        return self.main(x)
        

class UNetGenerator(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class Discriminator(nn.Module):
    def __init__(self, num_filters, num_bottleneck):
        super(Discriminator, self).__init__()
        
        layer = []
        layer.append(NormalBlock(3, num_filters))

        curr_num_filters = num_filters
        for i in range(num_bottleneck):
            layer.append(NormalBlock(curr_num_filters, curr_num_filters * 2))
            curr_num_filters = curr_num_filters * 2

        # layer.append(nn.Conv2d(curr_num_filters, 1, kernel_size=3, padding=1, bias=False))
        self.main = nn.Sequential(*layer)
        self.output_layer = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        out = out.view(out.shape[0], -1)
        validity = self.output_layer(out)
        validity = validity.view(validity.shape[0])
        del out
        return validity


