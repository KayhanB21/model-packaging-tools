# coding=utf-8
from config import settings

from tensorflow import keras
import numpy as np
import onnxruntime

(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

x_val = x_val.astype("float32") / 255
x_val = np.expand_dims(x_val, -1)
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

ort_session = onnxruntime.InferenceSession("./model/epoch_checkpoint.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: x_val}
ort_outs = ort_session.run(None, ort_inputs)

positive = np.sum(np.argmax(ort_outs[0], axis = 1) == y_val)
negative = np.sum(np.argmax(ort_outs[0], axis = 1) != y_val)

print(f"Positive cases = {positive}/{len(x_val)}, "
      f"Negative case = {negative}/{len(x_val)}, "
      f'validation accuracy % = {positive / (positive + negative) * 100:0.3f}')
