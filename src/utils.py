'''
Created on Novemver 26, 2017

@author: optas
'''

import tensorflow as tf


def leaky_relu(alpha):
    if not (alpha < 1 and alpha > 0):
        raise ValueError()

    return lambda x: tf.maximum(alpha * x, x)


def safe_log(x, eps=1e-12):
    return tf.log(tf.maximum(x, eps))