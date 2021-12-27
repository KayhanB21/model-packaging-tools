# coding=utf-8
from config import settings

import torch
import torchvision
import onnxruntime
import numpy as np

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
val_loader = torch.utils.data.DataLoader(dataset = torchvision.datasets.MNIST('./dataset/mnist/',
                                                                              train = False,
                                                                              download = True,
                                                                              transform = transform),
                                         batch_size = settings.training_setting.test_batch_size,
                                         shuffle = False)

ort_session = onnxruntime.InferenceSession("./model/epoch_checkpoint.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


positive = 0
negative = 0

for data, label in val_loader:
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
    ort_outs = ort_session.run(None, ort_inputs)
    positive += np.sum(np.argmax(ort_outs[0], axis = 1) == to_numpy(label))
    negative += np.sum(np.argmax(ort_outs[0], axis = 1) != to_numpy(label))

print(f"Positive cases = {positive}/{len(val_loader.dataset)}, "
      f"Negative case = {negative}/{len(val_loader.dataset)}, "
      f'validation accuracy % = {positive / (positive + negative) * 100:0.3f}')
