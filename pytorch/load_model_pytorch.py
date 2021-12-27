# coding=utf-8
from config import settings
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
val_loader = torch.utils.data.DataLoader(dataset = torchvision.datasets.MNIST('./dataset/mnist/',
                                                                              train = False,
                                                                              download = True,
                                                                              transform = transform),
                                         batch_size = settings.training_setting.test_batch_size,
                                         shuffle = False)


class MNISTClassifierNet(nn.Module):
    def __init__(self):
        super(MNISTClassifierNet, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels = 1,
                                  out_channels = 10,
                                  kernel_size = 5)
        self.conv2d_2 = nn.Conv2d(in_channels = 10,
                                  out_channels = 20,
                                  kernel_size = 5)
        self.drop2d_1 = nn.Dropout2d(p = 0.5)
        self.fc_1 = nn.Linear(in_features = 320, out_features = 150)
        self.fc_2 = nn.Linear(in_features = 150, out_features = 50)
        self.fc_3 = nn.Linear(in_features = 50, out_features = 10)

    def forward(self, x):
        # Updated 26 Dec, 2021
        # 512 is a batch size

        # conv1(x): 512*1*28*28 -> 512*10*(28-5[kernel_size]+1)*(28-5[kernel_size]+1) : 512*10*24*24
        # F.max_pool2d: 512*10*24*24 -> 512*10*(24/[kernel_size])*(24/[kernel_size]): 512*10*12*12
        x = F.relu(input = F.max_pool2d(input = self.conv2d_1(x), kernel_size = 2))
        # conv2(x): 512*10*12*12 -> 512*20*(12-5[kernel_size]+1)*(12-5[kernel_size]+1) : 512*20*8*8
        # F.max_pool2d: 512*20*8*8 -> 512*20*(8/[kernel_size])*(8/[kernel_size]): 512*20*4*4
        x = F.relu(input = F.max_pool2d(input = self.drop2d_1(self.conv2d_2(x)), kernel_size = 2))
        # x.view: 512*20*4*4 -> 512*320
        x = x.view(-1, 320)
        # fc1(x): 512*320 -> 512*150
        x = F.relu(input = self.fc_1(x))
        x = F.dropout(input = x, p = 0.5, training = self.training)
        # fc2(x): 512*150 -> 512*50
        x = F.relu(input = self.fc_2(x))
        x = F.dropout(input = x, p = 0.5, training = self.training)
        # fc3(x): 512*50 -> 512*10
        x = self.fc_3(x)
        # F.log_softmax: 512*10 -> 512*10
        return F.log_softmax(x, dim = -1)


net = MNISTClassifierNet()
checkpoint = torch.load('./model/epoch_checkpoint.pt')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

positive = 0
negative = 0

with torch.no_grad():
    for data, label in val_loader:
        prediction = net(data)
        positive += torch.sum(torch.argmax(prediction, dim = 1) == label)
        negative += torch.sum(torch.argmax(prediction, dim = 1) != label)

print(f"Positive cases = {positive}/{len(val_loader.dataset)}, "
      f"Negative case = {negative}/{len(val_loader.dataset)}, "
      f'validation accuracy % = {positive / (positive + negative) * 100:0.3f}')
