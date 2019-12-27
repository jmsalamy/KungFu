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

parser.add_argument('--epochs',
                    type=int,
                    default=15,
                    help='number of epochs')


parser.add_argument('--backup-frac',
                    type=float,
                    default=0.1,
                    help='fraction of total data sharded in the backup worker')

parser.add_argument('--dataset-size',
                    type=int,
                    default=50000,
                    help='size of the dataset for this test run')

parser.add_argument('--name',
                    type=str,
                    required=True,
                    help='name this experiement run for Tensorboard logging')

args = parser.parse_args()

# Model and dataset params
DATASET_SIZE = args.dataset_size
BACKUP_FRAC = args.backup_frac
BACKUP_WORKER_ID = 2
num_classes = 10
learning_rate = 0.01
batch_size = 128
epochs = args.epochs


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


def load_data_per_node(x_train, y_train, dataset_size, backup_worker_id, backup_frac=0.1):

    # custom size of the training data
    x_train, y_train = x_train[:DATASET_SIZE], y_train[:DATASET_SIZE]

    # shard the dataset for KungFu node
    n_shards = current_cluster_size()
    shard_id = current_rank()
    train_data_size = len(x_train)

    shard_size = train_data_size // n_shards
    offset = shard_size * shard_id

    # extract the data for learning of the KungFu primary nodes
    x_node = x_train[offset:offset + shard_size]
    y_node = y_train[offset:offset + shard_size]
    num_images_backup = int(train_data_size * backup_frac)

    # extract the data for learning of the KungFu backup nodes
    frac_data_per_worker = 1 / n_shards
    repeat_nums = frac_data_per_worker // backup_frac
    remainder = int(round(train_data_size *
                          (frac_data_per_worker - backup_frac*repeat_nums)))
    print("info : ", frac_data_per_worker, repeat_nums,
          backup_frac*repeat_nums, remainder, train_data_size)

    if shard_id == backup_worker_id:
        x_distinct = x_train[offset:offset + shard_size][0:num_images_backup]
        y_distinct = y_train[offset:offset + shard_size][0:num_images_backup]
        x_repeat = x_distinct.repeat(repeat_nums, axis=0)
        y_repeat = y_distinct.repeat(repeat_nums, axis=0)
        x_node = np.concatenate((x_repeat, x_distinct[0:remainder]), axis=0)
        y_node = np.concatenate((y_repeat, y_distinct[0:remainder]), axis=0)

    print("Worker ID {} | start idx {} | end idx {} ".format(
        shard_id, offset, offset+shard_size))
    print("Training set size:", x_node.shape, y_node.shape)

    return x_node, y_node


def train_model(model, model_name, x_train, x_test, y_train, y_test):
    # Pre-process dataset
    x_test = x_test.astype('float32')
    x_test /= 255
    x, y = load_data_per_node(
        x_train, y_train, DATASET_SIZE, backup_worker_id=BACKUP_WORKER_ID, backup_frac=BACKUP_FRAC)

    callbacks = [BroadcastGlobalVariablesCallback()]

    # Log to tensorboard for now
    if current_rank() == 0:
        logdir = "tensorboard-logs/{}/".format(model_name) + \
            datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)

    # Convert class vectors to binary class matrices.
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # train the model
    model.fit(x, y,
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
