# coding=utf-8
from config import settings

from tensorflow import keras
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

x_val = x_val.astype("float32") / 255
x_val = np.expand_dims(x_val, -1)
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

interpreter = tf.lite.Interpreter(model_path = './model/epoch_checkpoint.tflite')

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.resize_tensor_input(input_details[0]['index'], x_val.shape)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], x_val)

interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

val_acc_metric.update_state(y_val, prediction)
val_acc = val_acc_metric.result()
val_acc_metric.reset_states()
print(f"validation accuracy % = {val_acc * 100 :.3f}")
