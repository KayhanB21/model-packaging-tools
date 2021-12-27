# coding=utf-8
from config import settings

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import numpy as np
from tqdm import tqdm


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# making sure the random seed is set
if settings.training_setting.seed_random:
    seed_everything(seed = settings.training_setting.seed_random)

# instantiation of train and test dataloader
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# MNIST dataset that is normalized (0-1)
# data in each batch :512*1*28*28 (batch size, channels, image height, image width)
train_loader = torch.utils.data.DataLoader(dataset = torchvision.datasets.MNIST('./dataset/mnist/',
                                                                                train = True,
                                                                                download = True,
                                                                                transform = transform),
                                           batch_size = settings.training_setting.train_batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = torchvision.datasets.MNIST('./dataset/mnist/',
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
print(f"The network structure is: \n {net}")
print(f"Total parameters count is: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
print(f"Learning rate is: {settings.training_setting.learning_rate}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = net.parameters(),
                       lr = settings.training_setting.learning_rate)


def validation():
    net.eval()

    loss = 0
    positive = 0
    negative = 0

    with torch.no_grad():
        for data, label in test_loader:
            prediction = net(data)
            loss = criterion(prediction, label).item() * data.size(0)
            positive += torch.sum(torch.argmax(prediction, dim = 1) == label)
            negative += torch.sum(torch.argmax(prediction, dim = 1) != label)

    print(f"Validation, "
          f"Average loss = {loss / len(test_loader.dataset):.4f}, "
          f"Positive cases = {positive}/{len(test_loader.dataset)},"
          f"Negative case = {negative}/{len(test_loader.dataset)}, "
          f'validation accuracy % = {positive / (positive + negative) * 100:0.3f}')


# initial evaluation
validation()

# training loop
for epoch in range(settings.training_setting.max_epoch_count):
    print(f"Epoch: {epoch + 1}")
    train_loss = 0
    data_size_counter = 0

    net.train()

    for batch_id, (data, label) in enumerate(tqdm(train_loader)):

        optimizer.zero_grad()
        prediction = net(data)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        data_size_counter += data.size(0)

        if batch_id % settings.training_setting.log_batch_frequency == 0:
            print(f"Epoch: {epoch + 1}, batch: {batch_id + 1}/{len(train_loader)}, "
                  f"Average loss = {train_loss / data_size_counter:.4f}")

    print(f"Epoch: {epoch + 1}, batch: {batch_id + 1}/{len(train_loader)}, "
          f"Average loss = {train_loss / data_size_counter:.4f}")

    validation()

    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss},
               './model/epoch_checkpoint.pt')

    # saving jitted model
    with torch.no_grad():
        traced_cell = torch.jit.trace(net, data)
    torch.jit.save(traced_cell, './model/epoch_checkpoint_jit.pt')

    # saving onnx model
    torch.onnx.export(net,  # model being run
                      data,  # model input (or a tuple for multiple inputs)
                      './model/epoch_checkpoint.onnx',  # where to save the model (can be a file or file-like object)
                      export_params = True,  # store the trained parameter weights inside the model file
                      opset_version = 10,  # the ONNX version to export the model to
                      do_constant_folding = True,  # whether to execute constant folding for optimization
                      input_names = ['input'],  # the model's input names
                      output_names = ['output'],  # the model's output names
                      dynamic_axes = {'input': {0: 'batch_size'},  # variable length axes
                                      'output': {0: 'batch_size'}})
