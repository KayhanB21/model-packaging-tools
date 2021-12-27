# coding=utf-8
from config import settings

from tensorflow import keras
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

x_val = x_val.astype("float16") / 255
x_val = np.expand_dims(x_val, -1)
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

with tf.device("cpu:0"):
    reconstructed_model = keras.models.load_model("epoch_checkpoint", compile=False)
    prediction = reconstructed_model.predict(x_val)
    val_acc_metric.update_state(y_val, prediction)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print(f"validation accuracy % = {val_acc * 100 :.3f}")
