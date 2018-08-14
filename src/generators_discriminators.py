'''
Created on May 11, 2017

@author: optas
'''

import numpy as np
import tensorflow as tf
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout

from . encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only
from . tf_utils import leaky_relu
from . tf_utils import expand_scope_by_name


def mlp_discriminator(in_signal, non_linearity=tf.nn.relu, reuse=False, scope=None, b_norm=True, dropout_prob=None):
    ''' used in nips submission.
    '''
    encoder_args = {'n_filters': [64, 128, 256, 256, 512], 'filter_sizes': [1, 1, 1, 1, 1], 'strides': [1, 1, 1, 1, 1]}
    encoder_args['reuse'] = reuse
    encoder_args['scope'] = scope
    encoder_args['non_linearity'] = non_linearity
    encoder_args['dropout_prob'] = dropout_prob
    encoder_args['b_norm'] = b_norm
    layer = encoder_with_convs_and_symmetry(in_signal, **encoder_args)

    name = 'decoding_logits'
    scope_e = expand_scope_by_name(scope, name)
    d_logit = decoder_with_fc_only(layer, layer_sizes=[128, 64, 1], b_norm=b_norm, reuse=reuse, scope=scope_e)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def point_cloud_generator(z, pc_dims, layer_sizes=[64, 128, 512, 1024], non_linearity=tf.nn.relu, b_norm=False, b_norm_last=False, dropout_prob=None):
    ''' used in nips submission.
    '''
    
    n_points, dummy = pc_dims
    if (dummy != 3):
        raise ValueError()
    
    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm)
    out_signal = non_linearity(out_signal)
    
    if dropout_prob is not None:
        out_signal = dropout(out_signal, dropout_prob)

    if b_norm_last:
        out_signal = batch_normalization(out_signal)

    out_signal = fully_connected(out_signal, np.prod([n_points, 3]), activation='linear', weights_init='xavier')
    out_signal = tf.reshape(out_signal, [-1, n_points, 3])
    return out_signal


def convolutional_discriminator(in_signal, non_linearity=tf.nn.relu,
                                encoder_args={'n_filters': [128, 128, 256, 512], 'filter_sizes': [40, 20, 10, 10], 'strides': [1, 2, 2, 1]},
                                decoder_layer_sizes=[128, 64, 1],
                                reuse=False, scope=None):

    encoder_args['reuse'] = reuse
    encoder_args['scope'] = scope
    encoder_args['non_linearity'] = non_linearity
    layer = encoder_with_convs_and_symmetry(in_signal, **encoder_args)

    name = 'decoding_logits'
    scope_e = expand_scope_by_name(scope, name)
    d_logit = decoder_with_fc_only(layer, layer_sizes=decoder_layer_sizes, non_linearity=non_linearity, reuse=reuse, scope=scope_e)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def latent_code_generator(z, out_dim, layer_sizes=[64, 128], b_norm=False):
    layer_sizes = layer_sizes + out_dim
    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, b_norm=b_norm)
    out_signal = tf.nn.relu(out_signal)
    return out_signal


def latent_code_discriminator(in_singnal, layer_sizes=[64, 128, 256, 256, 512], b_norm=False, non_linearity=tf.nn.relu, reuse=False, scope=None):
    layer_sizes = layer_sizes + [1]
    d_logit = decoder_with_fc_only(in_singnal, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm, reuse=reuse, scope=scope)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def latent_code_discriminator_two_layers(in_signal, layer_sizes=[256, 512], b_norm=False, non_linearity=tf.nn.relu, reuse=False, scope=None):
    ''' Used in ICML submission.
    '''
    layer_sizes = layer_sizes + [1]
    d_logit = decoder_with_fc_only(in_signal, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm, reuse=reuse, scope=scope)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def latent_code_generator_two_layers(z, out_dim, layer_sizes=[128], b_norm=False):
    ''' Used in ICML submission.
    '''
    layer_sizes = layer_sizes + out_dim
    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, b_norm=b_norm)
    out_signal = tf.nn.relu(out_signal)
    return out_signal
