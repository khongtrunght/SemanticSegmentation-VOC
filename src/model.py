import torchvision
import torch
from torch import nn
from d2l import torch as d2l


num_classes = 21


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


def get_net(num_classes):
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    net = torch.nn.Sequential(*list(pretrained_net.children())[:-2])
    net.add_module('fc', torch.nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv', torch.nn.ConvTranspose2d(
        num_classes, num_classes, kernel_size=64, padding=16, stride=32))

    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)
    return net


if __name__ == '__main__':
    conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                    bias=False)
    conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

    img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
    X = img.unsqueeze(0)
    Y = conv_trans(X)
    out_img = Y[0].permute(1, 2, 0).detach()

    d2l.set_figsize()
    print('input image shape:', img.permute(1, 2, 0).shape)
    d2l.plt.imshow(img.permute(1, 2, 0))
    print('output image shape:', out_img.shape)
    d2l.plt.imshow(out_img)
    d2l.plt.show()
