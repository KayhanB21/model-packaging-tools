# coding=utf-8
from config import settings
import torch
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_loader = torch.utils.data.DataLoader(dataset = torchvision.datasets.MNIST('./dataset/mnist/',
                                                                               train = False,
                                                                               download = True,
                                                                               transform = transform),
                                          batch_size = settings.training_setting.test_batch_size,
                                          shuffle = False)

loaded_trace = torch.jit.load("./model/epoch_checkpoint_jit.pt")
with torch.no_grad():
    positive = 0
    negative = 0
    for data, label in test_loader:
        prediction = loaded_trace(data)
        positive += torch.sum(torch.argmax(prediction, dim = 1) == label)
        negative += torch.sum(torch.argmax(prediction, dim = 1) != label)
    print(f"Positive cases = {positive}/{len(test_loader.dataset)},"
          f"Negative case = {negative}/{len(test_loader.dataset)}, "
          f'validation accuracy % = {positive / (positive + negative) * 100:0.3f}')
