import numpy as np
from tensorflow.python.ops.init_ops import VarianceScaling

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def he_normal(seed=None):
    """He normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    Arguments:
        seed: A Python integer. Used to seed the random generator.
    Returns:
          An initializer.
    References:
          He et al., http://arxiv.org/abs/1502.01852
    Code:
        https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/keras/initializers.py
    """
    return VarianceScaling(scale=2., mode='fan_in', distribution='normal', seed=seed)


def img_unnorm(img, noise_type, noise_param):
    if noise_type == 'poisson':
        return (img + 0.5) * noise_param
    elif noise_type == 'gaussian':
        return img * 255
    else:
        raise NameError('No such noise type available!')


def save_sample_img(clean_list, ref_list, inf_list, noise_type, noise_param, colormap, img_path=None):
    list_len = len(clean_list)

    f = plt.figure()
    gs = gridspec.GridSpec(list_len, 3, wspace=0.0, hspace=0.0)

    for i in range(list_len):
        clean_img = img_unnorm(clean_list[i], noise_type, noise_param)

        ax = plt.subplot(gs[i, 0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(np.squeeze(clean_img), interpolation='nearest', cmap=colormap)

        ax = plt.subplot(gs[i, 1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(np.squeeze(ref_list[i]), interpolation='nearest', cmap=colormap)

        ax = plt.subplot(gs[i, 2])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(np.squeeze(inf_list[i]), interpolation='nearest', cmap=colormap)

    if img_path is not None:
        plt.savefig(img_path, dpi=200, bbox_inches='tight', pad_inches=0.0)
    plt.close(f)