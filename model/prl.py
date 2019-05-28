import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp
tfd = tfp.distributions

from utils.training_utils import he_normal


def conv_block(features,
               output_channels,
               kernel_shape,
               num_convs=2,
               initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
               regularizers=None,
               use_pooling=False,
               use_bias=False,
               use_batchnorm=False,
               is_training=False,
               use_nonlinearity=False,
               data_format='NCHW',
               name='conv_block'):
    """A convolutional block allows for pooling, convolutional, batch-norm and non-linearity layers"""

    with tf.variable_scope(name):
        if use_pooling:
            if data_format == 'NCHW':
                int_tuple = [1, 1, 2, 2]
            else:
                int_tuple = [1, 2, 2, 1]
            features = tf.nn.avg_pool(features, ksize=int_tuple, strides=int_tuple,
                                      padding='SAME', data_format=data_format)

        if not use_bias:
            initializers = {'w': initializers['w']}
        for _ in range(num_convs):
            features = snt.Conv2D(output_channels, kernel_shape, use_bias=use_bias, data_format=data_format,
                                  initializers=initializers, regularizers=regularizers)(features)
            if use_batchnorm:
                if data_format == 'NCHW':
                    bn_axis = 1
                else:
                    bn_axis = -1
                features = tf.layers.batch_normalization(features, axis=bn_axis, momentum=0.0, epsilon=1e-4,
                                                         training=is_training)
            if use_nonlinearity:
                features = tf.nn.relu(features)

        return features


class DnCNN(snt.AbstractModule):
    """An implementation of DnCNN for residual learning, at homo-dimensional level"""

    def __init__(self,
                 num_layers,
                 num_channels,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers=None,
                 data_format='NCHW',
                 name='dncnn'):
        super(DnCNN, self).__init__(name=name)
        self._num_layers = num_layers
        self._num_channels = num_channels
        self._initializers = initializers
        self._regularizers = regularizers
        self._data_format = data_format

    def _build(self, features, is_training):
        """one input block, several intermediate blocks and one output block"""
        features = conv_block(features, output_channels=64, kernel_shape=3, num_convs=1,
                              initializers=self._initializers, regularizers=self._regularizers,
                              use_bias=True, use_nonlinearity=True,
                              data_format=self._data_format, name='dncnn_input_block')

        features = conv_block(features, output_channels=64, kernel_shape=3, num_convs=self._num_layers-2,
                              initializers=self._initializers, regularizers=self._regularizers,
                              use_batchnorm=True, is_training=is_training, use_nonlinearity=True,
                              data_format=self._data_format, name='dncnn_intermediate_block')

        features = conv_block(features, output_channels=self._num_channels, kernel_shape=3, num_convs=1,
                              initializers=self._initializers, regularizers=self._regularizers,
                              data_format=self._data_format, name='dncnn_output_block')

        return features


class VGG_Encoder(snt.AbstractModule):
    """An implementation of quasi VGG-style CNN with M x (pooling, N x conv)-operations,
       where M = len(num_channels), N = num_convs_per_block, at low-dimensional level."""

    def __init__(self,
                 num_channels,
                 num_convs_per_block=3,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0), 'b': tf.contrib.layers.l2_regularizer(1.0)},
                 data_format='NCHW',
                 name='vgg_encoder'):
        super(VGG_Encoder, self).__init__(name=name)
        self._num_channels = num_channels
        self._num_convs = num_convs_per_block
        self._initializers = initializers
        self._regularizers = regularizers
        self._data_format = data_format

    def _build(self, inputs):
        features = [inputs]

        for i, n_channels in enumerate(self._num_channels):
            if i == 0:
                use_pooling = False
            else:
                use_pooling = True
            tf.logging.info('encoder scale {}: {}'.format(i, features[-1].get_shape()))
            features.append(conv_block(features[-1],
                                       output_channels=n_channels,
                                       kernel_shape=3,
                                       num_convs=self._num_convs,
                                       initializers=self._initializers,
                                       regularizers=self._regularizers,
                                       use_pooling=use_pooling,
                                       use_bias=True,
                                       use_nonlinearity=True,
                                       data_format=self._data_format,
                                       name='down_block_{}'.format(i)))

        return features[1:]


class DiagMultiGaussian(snt.AbstractModule):
    """A CNN outputting mean and variance for a diagonal multivariate Gaussian distribution"""

    def __init__(self,
                 latent_dim,
                 num_channels,
                 num_convs_per_block=3,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0), 'b': tf.contrib.layers.l2_regularizer(1.0)},
                 data_format='NCHW',
                 name='diag_gaussian'):
        self._latent_dim = latent_dim
        self._initializers = initializers
        self._regularizers = regularizers
        self._data_format = data_format

        if self._data_format == 'NCHW':
            self._channel_axis = 1
            self._spatial_axes = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_axes = [1, 2]

        super(DiagMultiGaussian, self).__init__(name=name)
        with self._enter_variable_scope():
            tf.logging.info('Building Conv Diagonal Gaussian.')
            self._encoder = VGG_Encoder(num_channels, num_convs_per_block, initializers, regularizers, data_format)

    def _build(self, img, gt=None):
        if gt is not None:
            img = tf.concat([img, gt], axis=self._channel_axis)
        encoding = self._encoder(img)[-1]
        encoding = tf.reduce_mean(encoding, axis=self._spatial_axes, keepdims=True)

        mu_log_sigma = snt.Conv2D(2*self._latent_dim, (1,1), data_format=self._data_format,
                                  initializers=self._initializers, regularizers=self._regularizers)(encoding)
        mu_log_sigma = tf.squeeze(mu_log_sigma, axis=self._spatial_axes)
        mu = mu_log_sigma[:, :self._latent_dim]
        log_sigma = mu_log_sigma[:, self._latent_dim:]

        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))





































