import tensorflow as tf

from importlib.machinery import SourceFileLoader
import argparse
import os
import numpy as np
import time
import logging
from tqdm import tqdm

from model.prl import PRL
import utils.training_utils as training_utils
from data.data_generator import test_data_list


def sample(cf, args):
    """Sampling from the learnt conditional distribution."""

    sample_size = args.sample_size
    time_stamp = args.time_stamp
    ckpt_dir = os.path.join(cf.project_dir, 'experiments', time_stamp)
    sample_dir = os.path.join(cf.project_dir, 'samples', time_stamp)

    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    log_path = os.path.join(sample_dir, 'sampling_stat.log')
    logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='a')

    prl_dncnn = PRL(latent_dim=cf.latent_dim,
                    output_channels=cf.output_channels,
                    num_channels=cf.num_channels,
                    det_net_depth=cf.det_net_depth,
                    merging_depth=cf.merging_depth,
                    num_convs_per_block=cf.num_convs_per_block,
                    initializers={'w': training_utils.he_normal(),
                                  'b': tf.truncated_normal_initializer(stddev=0.001)},
                    regularizers={'w': tf.contrib.layers.l2_regularizer(1.0)},
                    data_format=cf.data_format,
                    name='prl_dncnn')

    x = tf.placeholder(tf.float32, shape=cf.network_input_shape, name='observation')
    y = tf.placeholder(tf.float32, shape=cf.network_output_shape, name='ground_truth')
    is_training = tf.placeholder(tf.bool)

    prl_dncnn(x, y, is_training, is_inference=True)
    sampled_imgs = prl_dncnn.inference_sample(x)

    saver = tf.train.Saver()

    [val_data_noisy_list, val_data_clean_list] = test_data_list(img_dir=cf.validation_data_dir,
                                                                noise_type=cf.noise_type,
                                                                noise_param=cf.noise_param,
                                                                data_format=cf.data_format)
    num_data = len(val_data_clean_list)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        for i in tqdm(range(num_data)):
            restored_samples = []
            sampling_start_time = time.time()
            for j in range(sample_size):
                smpl_img = sess.run(sampled_imgs,
                                    feed_dict={x: val_data_noisy_list[i],
                                               is_training: False})
                restored_samples.append(smpl_img)
            sampling_time_delta = time.time() - sampling_start_time
            restored_samples = np.asarray(restored_samples)

            if cf.data_format == 'NCHW':
                restored_samples = np.squeeze(restored_samples, axis=(1, 2))
            else:
                restored_samples = np.squeeze(restored_samples, axis=(1, 4))

            save_path = os.path.join(sample_dir, '{}_img{}_t{}_s{}.npy'.format(cf.validation_data_name,
                                                                               i,
                                                                               time_stamp,
                                                                               sample_size))
            np.save(save_path, restored_samples)
            logging.info('{}s used for image {} of  sample size {}, average time: {} s/sample'.format
                         (sampling_time_delta, i, sample_size, sampling_time_delta/sample_size))

    val_noisy_path = os.path.join(sample_dir, '{}_val_noisy.npy'.format(cf.validation_data_name))
    val_clean_path = os.path.join(sample_dir, '{}_val_clean.npy'.format(cf.validation_data_name))

    for i in range(len(val_data_noisy_list)):
        val_data_noisy_list[i] = np.squeeze(val_data_noisy_list[i], axis=0)
        val_data_clean_list[i] = np.squeeze(val_data_clean_list[i], axis=0)

    np.save(val_noisy_path, np.asarray(val_data_noisy_list))
    np.save(val_clean_path, np.asarray(val_data_clean_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sampling of PRL with DnCNN')
    parser.add_argument('-c', '--config', type=str, default='./config/config_template.py',
                        help='path of the configuration file of this sampling')
    parser.add_argument('-t', '--time_stamp', type=str, default='0605_1620',
                        help='time stamp for a specified training')
    parser.add_argument('-s', '--sample_size', type=int, default=1000,
                        help='sample size of output images for each input image')
    args = parser.parse_args()

    cf = SourceFileLoader('cf', args.config).load_module()

    sample(cf, args)