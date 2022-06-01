from d2l import torch as d2l
import torch
import wandb


def train(net, epochs, train_iter, test_iter=None):
    net = net.cuda()

    def loss(inputs, target):
        return torch.nn.functional.cross_entropy(inputs, target, reduction='none').mean(dim=[1, 2])
    trainer = torch.optim.Adam(net.parameters())

    for epoch in range(epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for iter in train_iter:
            net.train()
            trainer.zero_grad()
            features, labels = iter

            features = features.cuda()
            labels = labels.cuda()

            outputs = net(features)
            loss_out = loss(outputs, labels)
            loss_out.sum().backward()
            trainer.step()
            train_loss_sum = loss_out.sum()
            train_acc_sum = d2l.accuracy(outputs, labels)
            metric.add(train_loss_sum, train_acc_sum,
                       labels.shape[0], labels.numel())
        print("Epoch : ", epoch, "Loss : ",
              metric[0] / metric[2], "Acc: ", metric[1]/metric[3])


class Trainer:
    def __init__(self, net, epochs, train_iter, test_iter=None, **kwargs):
        self.net = net
        self.epochs = epochs
        self.train_iter = train_iter
        self.test_iter = test_iter

        self.trainer = torch.optim.Adam(net.parameters())

        self.test_samples = next(iter(test_iter))
        self.test_images = self.test_samples[0][:min(test_iter.batch_size, 8)]
        self.test_labels = self.test_samples[1][:min(test_iter.batch_size, 8)]
        self.class_labels = kwargs.get('class_labels', None)

    def loss(self, inputs, target):
        return torch.nn.functional.cross_entropy(inputs, target, reduction='none').mean(dim=[1, 2])

    def train_step(self, iter):
        self.net.train()
        self.trainer.zero_grad()
        features, labels = iter

        features = features.cuda()
        labels = labels.cuda()

        outputs = self.net(features)
        loss_out = self.loss(outputs, labels)
        loss_out.sum().backward()
        self.trainer.step()
        train_loss_sum = loss_out.sum()
        train_acc_sum = d2l.accuracy(outputs, labels)
        return train_loss_sum, train_acc_sum, labels.shape[0], labels.numel()

    def train(self):
        for epoch in range(self.epochs):
            # Sum of training loss, sum of training accuracy, no. of examples,
            # no. of predictions
            metric = d2l.Accumulator(4)
            for iter in self.train_iter:
                train_loss_sum, train_acc_sum, num_examples, num_preds = self.train_step(
                    iter)
                metric.add(train_loss_sum, train_acc_sum,
                           num_examples, num_preds)
            self.on_epoch_end(epoch, metric=metric)

        return self.net

    def on_epoch_end(self, epoch, **kwargs):

        mask_list = []

        metric = kwargs['metric']

        print("Epoch : ", epoch, "Loss : ",
              metric[0] / metric[2], "Acc: ", metric[1]/metric[3])

        self.net.eval()

        pred_test = self.net(self.test_images.cuda()).argmax(dim=1)

        for i, (image, label) in enumerate(zip(self.test_images, self.test_labels)):
            pred_mask = pred_test[i].cpu().numpy()
            true_mask = label.cpu().numpy()
            mask_list.append(self.wb_mask(image.cpu(), pred_mask, true_mask))

        wandb.log({"Loss": metric[0] / metric[2],
                  "Accuracy": metric[1]/metric[3]}, step=epoch)
        wandb.log({"Prediction": mask_list}, step=epoch)

    def wb_mask(self, bg_img, pred_mask, true_mask):
        return wandb.Image(bg_img, masks={
            "prediction": {"mask_data": pred_mask, "class_labels": self.class_labels()},
            "ground truth": {"mask_data": true_mask, "class_labels": self.class_labels()}})

    def class_labels(self):
        return self.class_labels
