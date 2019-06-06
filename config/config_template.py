import os
import datetime
#############################################
#               Path settings               #
#############################################

# project root directory
project_dir = '/media/chen/Res/Python/prob_res_learning'

# timestamp
time_stamp = datetime.datetime.now().strftime('%m%d_%H%M')

# experiments directory
experiment_dir = os.path.join(project_dir, 'experiments', time_stamp)
experiment_image_dir = os.path.join(experiment_dir, 'epoch_imgs')

# samples directory
sample_dir = os.path.join(project_dir, 'samples', time_stamp)

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
cpu_device = '/cpu:0'
gpu_device = '/gpu:0'

regularization_weight = 1e-5
kl_weight = 0.1

num_training_batches = 279300
val_every_n_batches = 1862

learning_rate_schedule = 'piecewise_constant'
learning_rate_kwargs = {'values': [1e-3, 1e-3/10, 1e-5, 1e-6],
                        'boundaries': [55860, 111720, 148960],
                        'name': 'piecewise_constant_lr_decay'}
initial_learning_rate = learning_rate_kwargs['values'][0]

analytic_kl = True
use_ref_mean = False
save_every_n_steps = num_training_batches // 3 if num_training_batches >= 100000 else num_training_batches
disable_progress_bar = False

#############################################
#             Data generation               #
#############################################
data_format = 'NHWC'
every_n_epochs = 1862
batch_size = 128
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
merging_depth=5













