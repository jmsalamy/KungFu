"""KungFu experiment_0

KungFu requires users to make the following changes:
1. KungFu provides distributed optimizers that can wrap the original optimizer.
The distributed optimizer defines how local gradients and model weights are synchronized.
2. (Optional) In a distributed training setting, the training dataset is often partitioned.
3. (Optional) Scaling the learning rate of your local optimizer
"""

from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
from datetime import datetime

# kungfu imports
import kungfu as kf
import numpy as np
# tensorflow imports
import tensorflow as tf
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback
from kungfu.tensorflow.ops import broadcast
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)
# tf.keras imports
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, Input, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import preprocess_data
# local imports
from model_definition import Conv4_model

parser = argparse.ArgumentParser(
    description='CIFAR10 primary/backup Experiments')
parser.add_argument('--kf-optimizer',
                    type=str,
                    default='sync-sgd',
                    help='available options: sync-sgd, async-sgd, sma')
parser.add_argument('--name',
                    type=str,
                    required=True,
                    help='name this experiement run for Tensorboard logging')
args = parser.parse_args()

# Model and dataset params
num_classes = 10
learning_rate = 0.01
batch_size = 128
epochs = 20


def build_optimizer(name, n_workers=1):
    # Scale learning rate according to the level of data parallelism
    optimizer = tf.keras.optimizers.SGD(learning_rate=(learning_rate *
                                                       n_workers))

    # KUNGFU: Wrap the TensorFlow optimizer with KungFu distributed optimizers.
    if name == 'sync-sgd':
        return SynchronousSGDOptimizer(optimizer, use_locking=True)
    elif name == 'async-sgd':
        return PairAveragingOptimizer(optimizer)
    elif name == 'sma':
        return SynchronousAveragingOptimizer(optimizer)
    else:
        raise RuntimeError('unknown optimizer: %s' % name)


def build_model(optimizer, x_train, num_classes):
    model = Conv4_model(x_train, num_classes)
    # Compile model using kungfu optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def train_model(model, model_name, x_train, x_test, y_train, y_test):
    # Pre-process dataset
    x_test = x_test.astype('float32')
    x_test /= 255

    # Convert class vectors to binary class matrices.
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Train model
    n_workers = current_cluster_size()
    shard_id = current_rank()
    len_data = len(x_train)

    print("training set size:", x_train.shape, y_train.shape)
    # x_node, y_node = preprocess_data.data_shard(
    #     x_train, y_train, n_workers, shard_id, len_data)

    callbacks = [BroadcastGlobalVariablesCallback()]

    # Log to tensorboard for now
    if current_rank() == 0:
        logdir = "tensorboard-logs/{}/".format(model_name) + \
            datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=False,
              verbose=1,
              callbacks=callbacks)


def evaluate_trained_cifar10_model(model_name, x_test, y_test):
    model = load_model(model_file)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, num_classes)
    scores = model.evaluate(x_test, y_test, verbose=1)
    return scores


def f_data(x_train, y_train):
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)

    return x_train, y_train


if __name__ == "__main__":
    logging.basicConfig(filename="tf2_Conv4_CIFAR10_exp_0.log",
                        level=logging.DEBUG,
                        format="%(asctime)s:%(levelname)s:%(message)s")
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    class_names = ["airplane", "automobile", "bird", "cat",
                   "deer", "dog", "frog", "horse", "ship", "truck"]
    # Pre process data
    x_train, y_train = preprocess_data.process(f_data, x_train, y_train)

    optimizer = build_optimizer('sync-sgd', n_workers=current_cluster_size())
    model = build_model(optimizer, x_train, num_classes)
    train_model(model, args.name, x_train, x_test, y_train, y_test)
