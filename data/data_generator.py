import glob
import numpy as np
import os
from PIL import Image


# global variables for getting patches
patch_size, stride = 40, 10
scales = [1, 0.9, 0.8, 0.7]


def data_aug(img, mode=0):
    """
    Eight data augmentation schemes
    :param img:
    :param mode:
    :return:
    """
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def get_aug_patches(img_path, aug_times=1):
    """
    Read an image from img_path and return augmented patches
    :param img_path: string, path to the image
    :param aug_times: int, number of augmentation per patch
    :return patches: list, list of ndarrays of size (patch_size, patch_size)
    """
    img = Image.open(img_path)
    # the size of a PIL.Image is (width, height)
    w, h = img.size
    patches = []
    for s in scales:
        w_scaled, h_scaled = int(w * s), int(h * s)
        # of shape (height, width, 3) for RGB img, (height, width) for gray scale
        img_scaled = np.asarray(img.resize((w_scaled, h_scaled), Image.BICUBIC))
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    return patches


def gt_generator(img_dir, data_format, batch_size=128):
    """
    Generate ground truth images data
    :param img_dir:
    :param data_format:
    :param batch_size:
    :return gt_data:
    """
    img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    img_path_list += sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    img_path_list += sorted(glob.glob(os.path.join(img_dir, '*.JPEG')))

    gt_data = []
    for i in range(len(img_path_list)):
        patch = get_aug_patches(img_path_list[i])
        gt_data += patch
    gt_data = np.asarray(gt_data, dtype='uint8')
    discard_n = len(gt_data) - len(gt_data) // batch_size * batch_size
    gt_data = np.delete(gt_data, range(discard_n), axis=0)

    if gt_data.ndim == 3:
        gt_data = gt_data.reshape((gt_data.shape[0], gt_data.shape[1], gt_data.shape[2], 1))

    if data_format == 'NCHW':
        return np.transpose(gt_data, (0, 3, 1, 2))
    else:
        return gt_data


def add_noise(batch_x, noise_type, noise_param):
    """
    Return noisy images and clean images after normalisation
    :param batch_x:
    :param noise_type:
    :param noise_param:
    :return:
    """
    if noise_type == 'poisson':
        return np.random.poisson(batch_x * noise_param) / noise_param - 0.5, batch_x - 0.5
    elif noise_type == 'gaussian':
        return batch_x + np.random.normal(0, noise_param / 255.0, batch_x.shape), batch_x
    else:
        raise NameError('No such noise type available!')


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
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                # images in the range [0, 1]
                batch_x = xs[indices[i:i + batch_size]]
                # return normalised batch_y and batch_x
                yield add_noise(batch_x, noise_type, noise_param)

#
# # use_img_dir = './BSDS500/val'
# use_img_dir = './Test12'
# # data = gt_generator(use_img_dir)
# data_generator = train_generator(use_img_dir, data_format='NCHW', every_n_epochs=5, batch_size=128, noise_type='poisson', noise_param=1)
