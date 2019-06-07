import tensorflow as tf

from importlib.machinery import SourceFileLoader
import argparse
import os
import numpy as np
import time
import logging

from model.prl import PRL
import utils.training_utils as training_utils
from data.data_generator import test_data_list

parser = argparse.ArgumentParser(description='Sampling of PRL with DnCNN')
parser.add_argument('-c', '--config', type=str, default='./config/config_template.py',
                    help='path of the configuration file of this sampling')
parser.add_argument('-t', '--time_stamp', type=str, default='0605_1620',
                    help='time stamp for a specified training')
parser.add_argument('-s', '--sample_size', type=int, default=1000,
                    help='sample size of output images for each input image')
args = parser.parse_args()

cf = SourceFileLoader('cf', args.config).load_module()
sample_dir = os.path.join(cf.project_dir, 'samples', args.time_stamp)

[val_data_noisy_list, val_data_clean_list] = test_data_list(img_dir=cf.validation_data_dir,
                                                            noise_type=cf.noise_type,
                                                            noise_param=cf.noise_param,
                                                            data_format=cf.data_format)

val_noisy_path = os.path.join(sample_dir, '{}_val_noisy.npy'.format(cf.validation_data_name))
val_clean_path = os.path.join(sample_dir, '{}_val_clean.npy'.format(cf.validation_data_name))

for i in range(len(val_data_clean_list)):
    val_data_clean_list[i] = np.squeeze(val_data_clean_list[i], axis=0)
    val_data_noisy_list[i] = np.squeeze(val_data_noisy_list[i], axis=0)

np.save(val_noisy_path, val_data_noisy_list)
np.save(val_clean_path, val_data_clean_list)