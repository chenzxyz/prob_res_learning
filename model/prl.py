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
                 output_channels,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers=None,
                 data_format='NCHW',
                 name='dncnn'):
        super(DnCNN, self).__init__(name=name)
        self._num_layers = num_layers
        self._output_channels = output_channels
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

        features = conv_block(features, output_channels=self._output_channels, kernel_shape=3, num_convs=1,
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


class Merging(snt.AbstractModule):
    """Transform samples from low-dimensional space and homo-dimensional features into homo-dimensional samples;
       Subtract the residual samples from input and get the output samples"""

    def __init__(self,
                 num_layers,
                 output_channels,
                 initializers={'w': tf.orthogonal_initializer(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0), 'b': tf.contrib.layers.l2_regularizer(1.0)},
                 data_format='NCHW',
                 name='merging'):
        super(Merging, self).__init__(name=name)
        self._num_layers = num_layers
        self._output_channels = output_channels,
        self._initializers = initializers
        self._regularizers = regularizers
        self._data_format = data_format

        if self._data_format == 'NCHW':
            self._channel_axis = 1
            self._spatial_axes = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_axes = [1, 2]

    def _build(self, inputs, homo_dim_features, low_dim_features):
        shp = tf.shape(homo_dim_features)
        spatial_shape = [shp[axis] for axis in self._spatial_axes]
        multiples = [1] + spatial_shape
        multiples.insert(self._channel_axis, 1)

        if len(low_dim_features.get_shape()) == 2:
            low_dim_features = tf.expand_dims(low_dim_features, axis=2)
            low_dim_features = tf.expand_dims(low_dim_features, axis=2)

        broadcast_low_dim_features = tf.tile(low_dim_features, multiples)
        residual = tf.concat([homo_dim_features, broadcast_low_dim_features], axis=self._channel_axis)
        residual = conv_block(residual, output_channels=32, kernel_shape=(1,1), num_convs=self._num_layers,
                              initializers=self._initializers, regularizers=self._regularizers,
                              use_bias=True, use_nonlinearity=True, data_format=self._data_format)
        residual = snt.Conv2D(output_channels=self._output_channels, kernel_shape=(1, 1), data_format=self._data_format,
                              initializers=self._initializers, regularizers=self._regularizers)(residual)

        return tf.math.subtract(inputs, residual, name='output_layer')


class PRL(snt.AbstractModule):
    """A Probabilistic Residual Learning implementation with DnCNN as the deterministic feature net"""

    def __init__(self,
                 latent_dim,
                 output_channels,
                 num_channels,
                 det_net_depth,
                 merging_depth,
                 num_convs_per_block=3,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0), 'b': tf.contrib.layers.l2_regularizer(1.0)},
                 data_format='NCHW',
                 name='prob_res_learning'):
        super(PRL, self).__init__(name=name)

        with self._enter_variable_scope():
            self._deterministic_net = DnCNN(num_layers=det_net_depth,
                                            output_channels=output_channels,
                                            initializers=initializers,
                                            regularizers=None,
                                            data_format=data_format,
                                            name='deterministic_net')

            self._inference_net = DiagMultiGaussian(latent_dim=latent_dim,
                                                    num_channels=num_channels,
                                                    num_convs_per_block=num_convs_per_block,
                                                    initializers=initializers,
                                                    regularizers=regularizers,
                                                    data_format=data_format,
                                                    name='stochastic_net')

            self._reference_net = DiagMultiGaussian(latent_dim=latent_dim,
                                                    num_channels=num_channels,
                                                    num_convs_per_block=num_convs_per_block,
                                                    initializers=initializers,
                                                    regularizers=regularizers,
                                                    data_format=data_format,
                                                    name='reference_net')

            self._merging_net = Merging(num_layers=merging_depth,
                                        output_channels=output_channels,
                                        initializers=initializers,
                                        regularizers=regularizers,
                                        data_format=data_format,
                                        name='merging_net')

    def _build(self, inputs, ground_truth, is_training, is_inference):
        self._ground_truth = ground_truth

        if not is_inference:
            self._q = self._reference_net(inputs, ground_truth)

        self._p = self._inference_net(inputs)
        self._det_features = self._deterministic_net(input, is_training)

    def reference_sample(self, inputs, use_dist_mean=False, z_q=None):
        """use reference distribution to recover a sample, cannot be used for inference!"""
        if use_dist_mean:
            z_q = self._q.loc
        else:
            if z_q is None:
                z_q = self._q.sample()
        return self._merging_net(inputs, self._det_features, z_q)

    def inference_sample(self, inputs):
        """use stochastic net to recover a sample, can be used for inference!"""
        z_p = self._p.sample()
        return self._merging_net(inputs, self._det_features, z_p)

    def kl(self, analytic=True, z_q=None):
        """evaluate the KL term in the loss function"""
        if analytic:
            return tfd.kl_divergence(self._q, self._p)
        else:
            if z_q is None:
                z_q = self._q.sample()
            log_q = self._q.log_prob(z_q)
            log_p = self._p.log_prob(z_q)
            return log_q - log_p

    def loss_mini_batch(self, inputs, ground_truth, beta=0.1, analytic_kl=True, use_ref_mean=False, z_q=None):
        if z_q is None:
            z_q = self._q.sample()

        self._kl_val = tf.reduce_mean(self.kl(analytic_kl, z_q))
        self._ref_sample = self.reference_sample(inputs, use_dist_mean=use_ref_mean, z_q=z_q)
        batch_size = 2 * tf.cast(tf.shape(ground_truth)[0], tf.float32)
        self._rec_loss = tf.reduce_sum(tf.square(ground_truth - self._ref_sample)) / batch_size

        return self._rec_loss + beta * self._kl_val
