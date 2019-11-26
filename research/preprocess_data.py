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
