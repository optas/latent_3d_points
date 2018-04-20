'''
Created on April 14, 2018

@author: optas, vahe1994

'''

import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout
from . tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers


def discriminator_smpl(input,scope=None,reuse=False):

    weight_decay=0.001

    layer_sizes=[256,512,1]
    n_layers =len(layer_sizes)
    for i in xrange(0,len(layer_sizes)-1):
        if(i==0):
            layer = input 
        name = 'discriminator_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = fully_connected(layer, layer_sizes[i], activation='relu', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)    
    
    sigm = tf.nn.sigmoid(layer,name=name)
    return sigm, layer

def generator_smpl(input,n_output,reuse=False):


    reuse=False
    scope='generator'
    layer_sizes=[128,n_output]
    n_layers =len(layer_sizes)
    for i in xrange(0,len(layer_sizes)-1):
        if(i==0):
            layer = input 
        name = 'generator_fc_' + str(i)

        scope_i = scope + name

        layer = fully_connected(layer, layer_sizes[i], activation='relu', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)    
    name = 'generator_fc_' + str(len(layer_sizes))
    scope_i = scope + name  
    layer = fully_connected(layer, layer_sizes[n_layers - 1][0], activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)
    return layer