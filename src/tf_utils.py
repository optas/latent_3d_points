'''
Created on November 26, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np


def expand_scope_by_name(scope, name):
    """ expand tf scope by given name.
    """

    if isinstance(scope, basestring):
        scope += '/' + name
        return scope

    if scope is not None:
        return scope.name + '/' + name
    else:
        return scope


def replicate_parameter_for_all_layers(parameter, n_layers):
    if parameter is not None and len(parameter) != n_layers:
        if len(parameter) != 1:
            raise ValueError()
        parameter = np.array(parameter)
        parameter = parameter.repeat(n_layers).tolist()
    return parameter


def reset_tf_graph():
    ''' Reset's all variables of default-tf graph. Useful for jupyter.
    '''
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    
    
def leaky_relu(alpha):
    if not (alpha < 1 and alpha > 0):
        raise ValueError()

    return lambda x: tf.maximum(alpha * x, x)


def safe_log(x, eps=1e-12):
    return tf.log(tf.maximum(x, eps))