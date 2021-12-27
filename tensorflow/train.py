# coding=utf-8
from config import settings

import random
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm


# to force using cpu add CUDA_VISIBLE_DEVICES="" to env variables
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# making sure the random seed is set
if settings.training_setting.seed_random:
    seed_everything(seed = settings.training_setting.seed_random)

# the data, split between train and test sets
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
# number_of_images * height * width * channels
x_train = x_train.astype("float16") / 255
x_val = x_val.astype("float16") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_loader = train_loader.shuffle(buffer_size = settings.training_setting.train_batch_size * 10).batch(settings.training_setting.train_batch_size)

val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_loader = val_loader.batch(settings.training_setting.test_batch_size)


class MNISTClassifierNet:
    def __init__(self):
        inputs = keras.Input(shape = (28, 28, 1))
        conv2d_1 = layers.Conv2D(filters = 10,
                                 kernel_size = (5, 5),
                                 activation = "relu")(inputs)
        maxpool2d_1 = layers.MaxPooling2D(pool_size = (2, 2))(conv2d_1)
        conv2d_2 = layers.Conv2D(filters = 20, kernel_size = (5, 5))(maxpool2d_1)
        drop2d_1 = layers.SpatialDropout2D(rate = 0.5)(conv2d_2)
        maxpool2d_2 = layers.MaxPooling2D(pool_size = (2, 2))(drop2d_1)
        relu_1 = layers.ReLU()(maxpool2d_2)
        flat_1 = layers.Flatten()(relu_1)
        fc_1 = layers.Dense(units = 150, activation = "relu")(flat_1)
        drop_1 = layers.Dropout(rate = 0.5)(fc_1)
        fc_2 = layers.Dense(units = 50, activation = "relu")(drop_1)
        drop_2 = layers.Dropout(rate = 0.5)(fc_2)
        fc_3 = layers.Dense(units = 10, activation = "softmax")(drop_2)
        self.model = keras.Model(inputs = inputs, outputs = fc_3)

    def get_model(self):
        return self.model


net = MNISTClassifierNet().get_model()
net.summary()
optimizer = keras.optimizers.Adam(learning_rate = settings.training_setting.learning_rate)
print(f"Learning rate is: {settings.training_setting.learning_rate}")
# If you want to provide labels using one-hot representation,
# please use CategoricalCrossentropy loss,
# y_train = keras.utils.to_categorical(y_train, num_classes)
loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction = 'sum')
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

with tf.device("cpu:0"):
    @tf.function(jit_compile = True)
    def train_step(data, label):
        with tf.GradientTape() as tape:
            prediction = net(data, training = True)
            loss_value = loss_fn(y_true = label, y_pred = prediction)
        grads = tape.gradient(loss_value, net.trainable_weights)
        optimizer.apply_gradients(zip(grads, net.trainable_weights))
        return loss_value


    @tf.function(jit_compile = True)
    def valid_step(data, label):
        prediction = net(data, training = False)
        val_acc_metric.update_state(label, prediction)


    # initial evaluation
    for batch_id, (data, label) in enumerate(val_loader):
        valid_step(data = data, label = label)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()

    print(f"validation accuracy % = {val_acc * 100 :.3f}")

    # training loop
    for epoch in range(settings.training_setting.max_epoch_count):
        print(f"Epoch: {epoch + 1}")
        train_loss = 0
        data_size_counter = 0

        for batch_id, (data, label) in enumerate(tqdm(train_loader)):
            loss_value = train_step(data = data, label = label)
            train_loss += loss_value
            data_size_counter += len(data)

            if batch_id % settings.training_setting.log_batch_frequency == 0:
                print(f"Epoch: {epoch + 1}, batch: {batch_id + 1}/{len(train_loader)}, "
                      f"Average loss = {train_loss / data_size_counter:.4f}")

        print(f"Epoch: {epoch + 1}, batch: {batch_id + 1}/{len(train_loader)}, "
              f"Average loss = {train_loss / data_size_counter:.4f}")

        for data, label in val_loader:
            valid_step(data = data, label = label)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()

        print(f"validation accuracy % = {val_acc * 100 :.3f}")
        net.save("./model/epoch_checkpoint")

        # saving tflite
        converter = tf.lite.TFLiteConverter.from_saved_model("./model/epoch_checkpoint")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()
        with open('./model/epoch_checkpoint.tflite', 'wb') as f:
            f.write(tflite_model)
