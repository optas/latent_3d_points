'''
Created on February 2, 2017

@author: optas
'''

import warnings
import os.path as osp
import tensorflow as tf
import numpy as np

from tflearn import is_training

from . in_out import create_dir, pickle_data, unpickle_data
from . general_utils import apply_augmentations, iterate_in_chunks
from . neural_net import Neural_Net, MODEL_SAVER_ID


class Configuration():
    def __init__(self, n_input, encoder, decoder, encoder_args={}, decoder_args={},
                 training_epochs=200, batch_size=10, learning_rate=0.001, denoising=False,
                 saver_step=None, train_dir=None, z_rotate=False, loss='chamfer', gauss_augment=None,
                 saver_max_to_keep=None, loss_display_step=1, debug=False,
                 n_z=None, n_output=None, latent_vs_recon=1.0, consistent_io=None):

        # Parameters for any AE
        self.n_input = n_input
        self.is_denoising = denoising
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

        # Training related parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.gauss_augment = gauss_augment
        self.z_rotate = z_rotate
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs
        self.debug = debug

        # Used in VAE
        self.latent_vs_recon = np.array([latent_vs_recon], dtype=np.float32)[0]
        self.n_z = n_z

        # Used in AP
        if n_output is None:
            self.n_output = n_input
        else:
            self.n_output = n_output

        self.consistent_io = consistent_io

    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    def __str__(self):
        keys = self.__dict__.keys()
        vals = self.__dict__.values()
        index = np.argsort(keys)
        res = ''
        for i in index:
            if callable(vals[i]):
                v = vals[i].__name__
            else:
                v = str(vals[i])
            res += '%30s: %s\n' % (str(keys[i]), v)
        return res

    def save(self, file_name):
        pickle_data(file_name + '.pickle', self)
        with open(file_name + '.txt', 'w') as fout:
            fout.write(self.__str__())

    @staticmethod
    def load(file_name):
        return unpickle_data(file_name + '.pickle').next()


class AutoEncoder(Neural_Net):
    '''Basis class for a Neural Network that implements an Auto-Encoder in TensorFlow.
    '''

    def __init__(self, name, graph, configuration):
        Neural_Net.__init__(self, name, graph)
        self.is_denoising = configuration.is_denoising
        self.n_input = configuration.n_input
        self.n_output = configuration.n_output

        in_shape = [None] + self.n_input
        out_shape = [None] + self.n_output

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            if self.is_denoising:
                self.gt = tf.placeholder(tf.float32, out_shape)
            else:
                self.gt = self.x
   
    def partial_fit(self, X, GT=None):
        '''Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a denoising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        is_training(True, session=self.sess)
        try:
            if GT is not None:
                _, loss, recon = self.sess.run((self.train_step, self.loss, self.x_reconstr), feed_dict={self.x: X, self.gt: GT})
            else:
                _, loss, recon = self.sess.run((self.train_step, self.loss, self.x_reconstr), feed_dict={self.x: X})

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return recon, loss

    def reconstruct(self, X, GT=None, compute_loss=True):
        '''Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)'''
        if compute_loss:
            loss = self.loss
        else:
            loss = self.no_op

        if GT is None:
            return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X})
        else:
            return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X, self.gt: GT})

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})

    def interpolate(self, x, y, steps):
        ''' Interpolate between and x and y input vectors in latent space.
        x, y np.arrays of size (n_points, dim_embedding).
        '''
        in_feed = np.vstack((x, y))
        z1, z2 = self.transform(in_feed.reshape([2] + self.n_input))
        all_z = np.zeros((steps + 2, len(z1)))

        for i, alpha in enumerate(np.linspace(0, 1, steps + 2)):
            all_z[i, :] = (alpha * z2) + ((1.0 - alpha) * z1)

        return self.sess.run((self.x_reconstr), {self.z: all_z})

    def decode(self, z):
        if np.ndim(z) == 1:  # single example
            z = np.expand_dims(z, 0)
        return self.sess.run((self.x_reconstr), {self.z: z})

    def train(self, train_data, configuration, log_file=None, held_out_data=None):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in xrange(c.training_epochs):
            loss, duration = self._single_epoch_train(train_data, c)
            epoch = int(self.sess.run(self.increment_epoch))
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                checkpoint_path = osp.join(c.train_dir, MODEL_SAVER_ID)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)

            if c.exists_and_is_not_none('summary_step') and (epoch % c.summary_step == 0 or epoch - 1 == 0):
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)

            if held_out_data is not None and c.exists_and_is_not_none('held_out_step') and (epoch % c.held_out_step == 0):
                loss, duration = self._single_epoch_train(held_out_data, c, only_fw=True)
                print("Held Out Data :", 'forward time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('On Held_Out: %04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))
        return stats

    def evaluate(self, in_data, configuration, ret_pre_augmentation=False):
        n_examples = in_data.num_examples
        data_loss = 0.
        pre_aug = None
        if self.is_denoising:
            original_data, ids, feed_data = in_data.full_epoch_data(shuffle=False)
            if ret_pre_augmentation:
                pre_aug = feed_data.copy()
            if feed_data is None:
                feed_data = original_data
            feed_data = apply_augmentations(feed_data, configuration)  # This is a new copy of the batch.
        else:
            original_data, ids, _ = in_data.full_epoch_data(shuffle=False)
            feed_data = apply_augmentations(original_data, configuration)

        b = configuration.batch_size
        reconstructions = np.zeros([n_examples] + self.n_output)
        for i in xrange(0, n_examples, b):
            if self.is_denoising:
                reconstructions[i:i + b], loss = self.reconstruct(feed_data[i:i + b], original_data[i:i + b])
            else:
                reconstructions[i:i + b], loss = self.reconstruct(feed_data[i:i + b])

            # Compute average loss
            data_loss += (loss * len(reconstructions[i:i + b]))
        data_loss /= float(n_examples)

        if pre_aug is not None:
            return reconstructions, data_loss, np.squeeze(feed_data), ids, np.squeeze(original_data), pre_aug
        else:
            return reconstructions, data_loss, np.squeeze(feed_data), ids, np.squeeze(original_data)
        
    def embedding_at_tensor(self, dataset, conf, feed_original=True, apply_augmentation=False, tensor_name='bottleneck'):
        '''
        Observation: the NN-neighborhoods seem more reasonable when we do not apply the augmentation.
        Observation: the next layer after latent (z) might be something interesting.
        tensor_name: e.g. model.name + '_1/decoder_fc_0/BiasAdd:0'
        '''
        batch_size = conf.batch_size
        original, ids, noise = dataset.full_epoch_data(shuffle=False)

        if feed_original:
            feed = original
        else:
            feed = noise
            if feed is None:
                feed = original

        feed_data = feed
        if apply_augmentation:
            feed_data = apply_augmentations(feed, conf)

        embedding = []
        if tensor_name == 'bottleneck':
            for b in iterate_in_chunks(feed_data, batch_size):
                embedding.append(self.transform(b.reshape([len(b)] + conf.n_input)))
        else:
            embedding_tensor = self.graph.get_tensor_by_name(tensor_name)
            for b in iterate_in_chunks(feed_data, batch_size):
                codes = self.sess.run(embedding_tensor, feed_dict={self.x: b.reshape([len(b)] + conf.n_input)})
                embedding.append(codes)

        embedding = np.vstack(embedding)
        return feed, embedding, ids
        
    def get_latent_codes(self, pclouds, batch_size=100):
        ''' Convenience wrapper of self.transform to get the latent (bottle-neck) codes for a set of input point 
        clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
        '''
        latent_codes = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            latent_codes.append(self.transform(pclouds[b]))
        return np.vstack(latent_codes)
