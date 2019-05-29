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