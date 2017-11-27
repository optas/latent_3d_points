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
            # g = tf.Graph()
            # with g.as_default():
        self.graph = graph
        self.name = name

        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

    def is_training(self):
        is_training_op = self.graph.get_collection('is_training')
        return self.sess.run(is_training_op)[0]

#     def __init__(self, name, model, trainer, sess):
#         '''
#         Constructor
#         '''
#         self.model = model
#         self.trainer = trainer
#         self.sess = sess
#         self.train_step = trainer.train_step
#         self.saver = tf.train.Saver(tf.global_variables(), scope=name, max_to_keep=None)
# 
#     def total_loss(self):
#         return self.trainer.total_loss
# 
#     def forward(self, input_tensor):
#         return self.model.forward(input_tensor)
# 
#     def save_model(self, tick):
#         self.saver.save(self.sess, MODEL_SAVER_ID, global_step=tick)
# 
#     def restore_model(self, model_path, tick, verbose=False):
#         ''' restore_model.
# 
#             Restore all the variables of the saved model.
#         '''
#         self.saver.restore(self.sess, osp.join(model_path, MODEL_SAVER_ID + '-' + str(int(tick))))