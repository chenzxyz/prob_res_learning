import numpy as np

import config.config_template as cf
from data.data_generator import gt_generator, add_noise

def train_generator(img_dir, data_format, every_n_epochs=5, batch_size=128, noise_type='poisson', noise_param=1):
    """
    Python generator for data batch (y, x)
    :param img_dir:
    :param data_format:
    :param every_n_epochs: regenerate clean image patches every n epochs
    :param batch_size:
    :param noise_type:
    :param noise_param:
    :return batch_y, batch_x:
    """
    while True:
        xs = gt_generator(img_dir, data_format, batch_size=batch_size)
        xs = xs.astype('float32') / 255.0
        indices = list(range(xs.shape[0]))
        for _ in range(every_n_epochs):
            print(len(indices))
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                # images in the range [0, 1]
                batch_x = xs[indices[i:i + batch_size]]
                # return normalised batch_y and batch_x
                yield add_noise(batch_x, noise_type, noise_param)


train_dataset = train_generator(img_dir=cf.training_data_dir, data_format=cf.data_format,
                                every_n_epochs=cf.shuffle_every_n_epochs, batch_size=cf.batch_size,
                                noise_type=cf.noise_type, noise_param=cf.noise_param)


xs = gt_generator(cf.training_data_dir, cf.data_format, batch_size=cf.batch_size)
xs = xs.astype('float32') / 255.0
indices = list(range(xs.shape[0]))