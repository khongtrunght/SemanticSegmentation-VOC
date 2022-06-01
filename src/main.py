from dataset import load_data_voc
from model import get_net
from train import Trainer
from utils import VOC_CLASSES
import torch
import wandb


def main():
    batch_size = 32
    crop_size = (320, 480)
    train_iter, test_iter = load_data_voc(batch_size, crop_size)
    class_labels = {i: k for i, k in enumerate(VOC_CLASSES)}

    net = get_net(num_classes=21)
    net.cuda()

    wandb.init(project="segmentation")
    train = Trainer(net, 10, train_iter, test_iter, class_labels=class_labels)
    train.train()

    torch.save(net.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
