'''
Created on April 27, 2017

@author: optas
'''
import numpy as np
import time
import tensorflow as tf

from . gan import GAN

from .. fundamentals.layers import safe_log
from tflearn import is_training


class LatentGAN(GAN):
    def __init__(self, name, learning_rate, n_output, noise_dim, discriminator, generator, beta=0.9, gen_kwargs={}, disc_kwargs={}, graph=None):

        self.noise_dim = noise_dim
        self.n_output = n_output
        self.discriminator = discriminator
        self.generator = generator

        GAN.__init__(self, name, graph)

        with tf.variable_scope(name):

            self.noise = tf.placeholder(tf.float32, shape=[None, noise_dim])                  # Noise vector.
            self.gt_data = tf.placeholder(tf.float32, shape=[None] + self.n_output)           # Ground-truth.

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output, **gen_kwargs)

            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.gt_data, scope=scope, **disc_kwargs)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, reuse=True, scope=scope, **disc_kwargs)

            self.loss_d = tf.reduce_mean(-tf.log(self.real_prob) - tf.log(1 - self.synthetic_prob))
            self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))

            #Post ICLR TRY: safe_log

            train_vars = tf.trainable_variables()

            d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]

            self.opt_d = self.optimizer(learning_rate, beta, self.loss_d, d_params)
            self.opt_g = self.optimizer(learning_rate, beta, self.loss_g, g_params)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.init = tf.global_variables_initializer()

            # Launch the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def _single_epoch_train(self, train_data, batch_size, noise_params):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        n_examples = train_data.num_examples
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        batch_size = batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        is_training(True, session=self.sess)
        try:
            # Loop over all batches
            for _ in xrange(n_batches):
                feed, _, _ = train_data.next_batch(batch_size)

                # Update discriminator.
                z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                feed_dict = {self.gt_data: feed, self.noise: z}
                loss_d, _ = self.sess.run([self.loss_d, self.opt_d], feed_dict=feed_dict)
                loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict=feed_dict)

                # Compute average loss
                epoch_loss_d += loss_d
                epoch_loss_g += loss_g

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)

        epoch_loss_d /= n_batches
        epoch_loss_g /= n_batches
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
