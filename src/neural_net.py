'''
Created on August 28, 2017

@author: optas
'''

import os.path as osp
import tensorflow as tf

MODEL_SAVER_ID = 'models.ckpt'


class Neural_Net(object):

    def __init__(self, name, graph):
        if graph is None:
            graph = tf.get_default_graph()

        self.graph = graph
        self.name = name

        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)
            self.increment_epoch = self.epoch.assign_add(tf.constant(1.0))

        self.no_op = tf.no_op()

    def is_training(self):
        is_training_op = self.graph.get_collection('is_training')
        return self.sess.run(is_training_op)[0]
    
    def restore_model(self, model_path, epoch, verbose=False):
        '''Restore all the variables of a saved model.
        '''
        self.saver.restore(self.sess, osp.join(model_path, MODEL_SAVER_ID + '-' + str(int(epoch))))

        if self.epoch.eval(session=self.sess) != epoch:
            warnings.warn('Loaded model\'s epoch doesn\'t match the requested one.')
        else:
            if verbose:
                print('Model restored in epoch {0}.'.format(epoch))
