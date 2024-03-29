import os
import datetime
#############################################
#               Path settings               #
#############################################

# project root directory
project_dir = 'CHANGE IT TO YOUR PROJECT DIRECTORY'

# timestamp
time_stamp = datetime.datetime.now().strftime('%m%d_%H%M')

# experiments directory
experiment_dir = os.path.join(project_dir, 'experiments', time_stamp)
experiment_image_dir = os.path.join(experiment_dir, 'epoch_imgs')

# configuration path
config_path = os.path.realpath(__file__)

# data
training_data_name = 'Train400'
validation_data_name = 'Test12'
training_data_dir = os.path.join(project_dir, 'data', training_data_name)
validation_data_dir = os.path.join(project_dir, 'data', validation_data_name)
#############################################
#       Settings on general training        #
#############################################
use_single_gpu = False
cuda_visible_devices = '0'
# cpu_device = '/cpu:0'
# gpu_device = '/gpu:0'

regularization_weight = 1e-5
kl_weight = 0.1

# Settings specifically for training dataset in use ---------
# Images after augmentation: 238336
# Batch size: 128
# Batches per epoch: 238336 / 128 = 1862
# Train for 279300 / 1862 = 150 epochs
# lr schedule boundary [30, 60, 80]
shuffle_every_n_epochs = 5
batch_size = 128
batches_per_epoch = 1862
num_training_batches = 279300

learning_rate_schedule = 'piecewise_constant'
learning_rate_kwargs = {'values': [1e-3, 1e-3/10, 1e-5, 1e-6],
                        'boundaries': [55860, 111720, 148960],
                        'name': 'piecewise_constant_lr_decay'}
initial_learning_rate = learning_rate_kwargs['values'][0]
# -----------------------------------------------------------

analytic_kl = True
use_ref_mean = False
save_every_n_steps = num_training_batches // 3 if num_training_batches >= 100000 else num_training_batches
disable_progress_bar = False

#############################################
#             Data generation               #
#############################################
data_format = 'NCHW'

noise_type = 'poisson'
noise_param = 1
output_channels = 1
if noise_type == 'poisson':
    peak_val = noise_param
elif noise_type == 'gaussian':
    peak_val = 255

if output_channels == 1:
    colormap = 'gray'
elif output_channels == 3:
    colormap = 'rgb'
#############################################
#             Network settings              #
#############################################
if data_format == 'NCHW':
    network_input_shape = (None, output_channels, None, None)
    network_output_shape = (None, output_channels, None, None)
else:
    network_input_shape = (None, None, None, output_channels)
    network_output_shape = (None, None, None, output_channels)
# homo-dimensional level
det_net_depth=17

# low-dimensional level
latent_dim = 6
num_convs_per_block=3
base_channels = 32
num_channels = [base_channels, 2*base_channels, 4*base_channels,
                6*base_channels, 6*base_channels, 6*base_channels, 6*base_channels]

# merging level
merging_depth=6













