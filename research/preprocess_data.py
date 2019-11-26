import numpy as np
import tensorflow as tf


def modify_image(f, image):
    """ Apply the function 'f' on 'image' and return the resulting image

    Args:
        f (function): function to be applied on image
        image (np.ndarray): image to be modified

    Returns:
        np.ndarray: modified image
    """

    modified_image = f(image)
    assert type(modified_image) == np.ndarray

    return modified_image


def modify_dataset(f, data, labels):
    """modify dataset by applying 'f' on data and labels

    Args:
        f (function): function to be applied on the data batch
        data (numpy.ndarray): array of data batch 
        labels (numpy.ndarray): array of corresponding labels

    Returns:
        numpy.ndarray: modified data and labels array
    """

    new_idx = f(data)
    modified_data = data[new_idx]
    modified_labels = labels[new_idx]

    return modified_data, modified_labels
