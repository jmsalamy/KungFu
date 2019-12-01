import numpy as np
import tensorflow as tf


def process(f_data, x_train, y_train):
    """ apply a user defined 'f_data' transormation on dataset

    Args:
        f_data (function): transformation function to apply on dataset
        x_train (numpy.ndarray): images
        y_train (numpy.ndarray): labels

    Returns:
        [numpy.ndarray]: transformed images and labels
    """

    x_train, y_train = f_data(x_train, y_train)
    assert type(x_train) == np.ndarray
    assert type(y_train) == np.ndarray

    return x_train, y_train


def data_shard(x_train, y_train, n_shards, shard_id, len_data, custom_data_binding={}, sharding_mode="equal"):
    """ shard data into available workers equally or via custom mapping

    Args:    
        x_train (numpy.ndarray) : training data
        y_train (numpy.ndarray) : training labels
        n_shards (int): number of workers/shards available
        shard_id (int): id of current shard
        len_data (int): length of data (rows) P
        custom_data_binding (dict, optional): custom mapping of data : worker. Defaults to {}.
        sharding_mode (str, optional): equal sharding vs custom. Defaults to "equal".

    Returns:
        [numpy.ndarray]: training data and labels for the 'shard_id' worker
    """

    if sharding_mode == "equal":

        shard_size = len_data // n_shards
        data_batch_size = len_data // n_shards
        offset = data_batch_size * shard_id

        # extract the data for learning of the KungFu node
        print("sharding info for current worker : ",
              shard_id, offset, offset + shard_size)
        x_node = x_train[offset:offset + shard_size]
        y_node = y_train[offset:offset + shard_size]

    if sharding_mode == "custom":
        if len(custom_data_binding) == 0:
            pass
        else:
            # implement custom binding here
            pass

    return x_node, y_node
