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
    elif arch == 'unet':
        net = Unet(3, num_classes)
    else:
        net = UnetPretrain(3, num_classes)

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


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class UnetPretrain(nn.Module):
    def __init__(self, in_channels, n_class):
        super(UnetPretrain, self).__init__()
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, images):
        x_original = self.conv_original_size0(images)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(images)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


if __name__ == '__main__':
    image = torch.randn(1, 3, 512, 512)

    net = UnetPretrain(3, 21)
    print(net(image).shape)
