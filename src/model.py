import torchvision
import torch
from torch import nn
from d2l import torch as d2l


num_classes = 21


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


def crop_images(imgs, target_imgs):
    target_width, target_height = target_imgs.shape[2:]
    height, width = imgs.shape[2:]
    x1 = (width - target_width) // 2
    y1 = (height - target_height) // 2
    return imgs[:, :, y1:y1+target_height, x1:x1+target_width]


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


def get_net(num_classes, arch='normal'):
    if arch == 'normal':
        pretrained_net = torchvision.models.resnet18(pretrained=True)
        net = torch.nn.Sequential(*list(pretrained_net.children())[:-2])
        net.add_module('conv', torch.nn.Conv2d(
            512, num_classes, kernel_size=1))
        net.add_module('transpose_conv', torch.nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, padding=16, stride=32))
        W = bilinear_kernel(num_classes, num_classes, 64)
        net.transpose_conv.weight.data.copy_(W)

    else:
        net = Unet(3, num_classes)

    return net


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(in_channels, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)
        self.up_tranv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_tranv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_tranv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_tranv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(1024, 512)
        self.up_conv2 = double_conv(512, 256)
        self.up_conv3 = double_conv(256, 128)
        self.up_conv4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, images):

        x1 = self.down_conv1(images)

        x2 = self.max_pool(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool(x7)
        x9 = self.down_conv5(x8)

        # decoder
        y = self.up_tranv1(x9)
        y1 = torch.cat([y, crop_images(x7, y)], dim=1)
        y2 = self.up_conv1(y1)

        y = self.up_tranv2(y2)
        y1 = torch.cat([y, crop_images(x5, y)], dim=1)
        y2 = self.up_conv2(y1)

        y = self.up_tranv3(y2)
        y1 = torch.cat([y, crop_images(x3, y)], dim=1)
        y2 = self.up_conv3(y1)

        y = self.up_tranv4(y2)
        y1 = torch.cat([y, crop_images(x1, y)], dim=1)
        y2 = self.up_conv4(y1)

        y = self.out(y2)
        return y


if __name__ == '__main__':
    image = torch.randn(1, 3, 256, 256)
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    net = torch.nn.Sequential(*list(pretrained_net.children())[:-2])
    out = net(image)
    print(out.shape)
