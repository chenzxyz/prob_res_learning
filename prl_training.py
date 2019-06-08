import os
import time
import shutil
import logging
import argparse
from importlib.machinery import SourceFileLoader

import tensorflow as tf
from tqdm import tqdm

from model.prl import PRL
import utils.training_utils as training_utils
from data.data_generator import train_generator, test_data_list


def train(cf):
    """Perform training from scratch."""

    if cf.use_single_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cf.cuda_visible_devices

    train_dataset = train_generator(img_dir=cf.training_data_dir, data_format=cf.data_format,
                                    every_n_epochs=cf.every_n_epochs, batch_size=cf.batch_size,
                                    noise_type=cf.noise_type, noise_param=cf.noise_param)

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
    beta = tf.placeholder(tf.float32, shape=(), name='beta')
    is_training = tf.placeholder(tf.bool)

    global_step = tf.train.get_or_create_global_step()

    if cf.learning_rate_schedule == 'piecewise_constant':
        learning_rate = tf.train.piecewise_constant(x=global_step, **cf.learning_rate_kwargs)
    else:
        learning_rate = tf.train.exponential_decay(learning_rate=cf.initial_learning_rate,
                                                   global_step=global_step,
                                                   **cf.learning_rate_kwargs)

    prl_dncnn(x, y, is_training, is_inference=False)

    model_loss = prl_dncnn.loss_mini_batch(x, y, beta=beta, analytic_kl=cf.analytic_kl, use_ref_mean=cf.use_ref_mean)
    reg_loss = cf.regularization_weight * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = model_loss + reg_loss

    # prepare for summaries
    ref_rec_loss = prl_dncnn._rec_loss
    kl = prl_dncnn._kl_val

    ref_sample = prl_dncnn._ref_sample
    inf_sample = prl_dncnn.inference_sample(x)

    gt_unnormed = training_utils.img_unnorm(y, cf.noise_type, cf.noise_param)
    ref_unnormed = training_utils.img_unnorm(ref_sample, cf.noise_type, cf.noise_param)
    inf_unnormed = training_utils.img_unnorm(inf_sample, cf.noise_type, cf.noise_param)

    if cf.data_format == 'NCHW':
        gt_unnormed = tf.transpose(gt_unnormed, perm=(0, 2, 3, 1))
        ref_unnormed = tf.transpose(ref_unnormed, perm=(0, 2, 3, 1))
        inf_unnormed = tf.transpose(inf_unnormed, perm=(0, 2, 3, 1))

    # can be used for both training and validation
    # ref_avg_mse = tf.reduce_mean(tf.metrics.mean_squared_error(gt_unnormed, ref_unnormed))
    # ref_avg_ssim = tf.reduce_mean(tf.image.ssim(gt_unnormed, ref_unnormed, max_val=cf.peak_val))
    # ref_avg_psnr = tf.reduce_mean(tf.image.psnr(gt_unnormed, ref_unnormed, max_val=cf.peak_val))
    inf_avg_mse = tf.reduce_mean(tf.metrics.mean_squared_error(gt_unnormed, inf_unnormed))
    inf_avg_ssim = tf.reduce_mean(tf.image.ssim(gt_unnormed, inf_unnormed, max_val=cf.peak_val))
    inf_avg_psnr = tf.reduce_mean(tf.image.psnr(gt_unnormed, inf_unnormed, max_val=cf.peak_val))

    # ---------------------------- training summaries
    # loss summaries
    train_ref_rec_loss_summary = tf.summary.scalar('train_ref_rec_loss', ref_rec_loss)
    train_kl_summary = tf.summary.scalar('train_kl', kl)
    train_model_loss_summary = tf.summary.scalar('train_elbo', model_loss)
    train_reg_loss_summary = tf.summary.scalar('train_reg_loss', reg_loss)
    train_loss_summary = tf.summary.scalar('train_loss', loss)
    # quantitative indicator summaries (not in use during training, to save time)
    # train_ref_avg_mse_summary = tf.summary.scalar('train_reference_avg_mse', ref_avg_mse)
    # train_ref_avg_ssim_summary = tf.summary.scalar('train_reference_avg_ssim', ref_avg_ssim)
    # train_ref_avg_psnr_summary = tf.summary.scalar('train_reference_avg_psnr', ref_avg_psnr)
    # train_inf_avg_mse_summary = tf.summary.scalar('train_reference_avg_mse', inf_avg_mse)
    # train_inf_avg_ssim_summary = tf.summary.scalar('train_reference_avg_ssim', inf_avg_ssim)
    # train_inf_avg_psnr_summary = tf.summary.scalar('train_reference_avg_psnr', inf_avg_psnr)
    # hyper-parameter summaries
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)
    beta_summary = tf.summary.scalar('kl_beta', beta)
    # merging summaries
    train_summary_op = tf.summary.merge([lr_summary, beta_summary,
                                         train_loss_summary,
                                         train_model_loss_summary,
                                         train_reg_loss_summary,
                                         train_ref_rec_loss_summary,
                                         train_kl_summary])

    # ---------------------------- timing summaries
    batches_per_second = tf.placeholder(tf.float32, shape=(), name='batches_per_second')
    timing_summary = tf.summary.scalar('batches_per_sec', batches_per_second)

    # ---------------------------- validation summaries
    val_avg_ref_rec_loss = tf.placeholder(tf.float32, shape=(), name='mean_val_ref_rec_loss')
    val_avg_kl = tf.placeholder(tf.float32, shape=(), name='mean_val_kl')
    # val_avg_ref_mse = tf.placeholder(tf.float32, shape=(), name='mean_val_ref_mse')
    # val_avg_ref_ssim = tf.placeholder(tf.float32, shape=(), name='mean_val_ref_ssim')
    # val_avg_ref_psnr = tf.placeholder(tf.float32, shape=(), name='mean_val_ref_psnr')
    val_avg_inf_mse = tf.placeholder(tf.float32, shape=(), name='mean_val_inf_mse')
    val_avg_inf_ssim = tf.placeholder(tf.float32, shape=(), name='mean_val_inf_ssim')
    val_avg_inf_psnr = tf.placeholder(tf.float32, shape=(), name='mean_val_inf_psnr')

    val_ref_rec_loss_summary = tf.summary.scalar('validation_ref_rec_loss', val_avg_ref_rec_loss)
    val_kl_summary = tf.summary.scalar('valiation_kl', val_avg_kl)
    val_avg_inf_mse_summary = tf.summary.scalar('validation_avg_inf_mse', val_avg_inf_mse)
    val_avg_inf_ssim_summary = tf.summary.scalar('validation_avg_inf_ssim', val_avg_inf_ssim)
    val_avg_inf_psnr_summary = tf.summary.scalar('validation_avg_inf_psnr', val_avg_inf_psnr)

    validation_summary_op = tf.summary.merge([val_ref_rec_loss_summary,
                                              val_kl_summary,
                                              val_avg_inf_mse_summary,
                                              val_avg_inf_ssim_summary,
                                              val_avg_inf_psnr_summary])

    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    tf.global_variables_initializer()

    saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=cf.experiment_dir,
                                              save_steps=cf.save_every_n_steps,
                                              saver=tf.train.Saver(save_relative_paths=True))

    shutil.copyfile(cf.config_path, os.path.join(cf.experiment_dir, 'used_config.py'))

    with tf.train.MonitoredTrainingSession(hooks=[saver_hook]) as sess:
        summary_writer = tf.summary.FileWriter(cf.experiment_dir, sess.graph)
        logging.info('Model: {}'.format(cf.experiment_dir))

        training_start_time = time.time()
        for i in tqdm(range(cf.num_training_batches), disable=cf.disable_progress_bar):
            sess_start_time = time.time()
            [train_data_noisy, train_data_clean] = next(train_dataset)
            _, train_summary = sess.run([optimizer, train_summary_op],
                                        feed_dict={x: train_data_noisy, y: train_data_clean,
                                                   beta: cf.kl_weight, is_training: True})
            summary_writer.add_summary(train_summary, i)
            sess_time_delta = time.time() - sess_start_time

            train_speed = sess.run(timing_summary, feed_dict={batches_per_second: 1. / sess_time_delta})
            summary_writer.add_summary(train_speed, i)

            if i % cf.val_every_n_batches == 0:
                running_avg_val_ref_rec_loss = 0.
                running_avg_val_kl = 0.
                running_avg_val_inf_mse = 0.
                running_avg_val_inf_ssim = 0.
                running_avg_val_inf_psnr = 0.

                [val_data_noisy_list, val_data_clean_list] = test_data_list(img_dir=cf.validation_data_dir,
                                                                            noise_type=cf.noise_type,
                                                                            noise_param=cf.noise_param,
                                                                            data_format=cf.data_format)
                num_val_data = len(val_data_clean_list)
                val_ref_img_list = []
                val_inf_img_list = []
                for j in range(num_val_data):
                    val_ref_img, val_inf_img, val_ref_rec_loss, val_kl, val_inf_mse, val_inf_ssim, val_inf_psnr = \
                        sess.run([ref_unnormed, inf_unnormed,
                                  ref_rec_loss, kl, inf_avg_mse, inf_avg_ssim, inf_avg_psnr],
                                 feed_dict={x: val_data_noisy_list[j], y: val_data_clean_list[j],
                                            beta: cf.kl_weight, is_training: False})

                    running_avg_val_ref_rec_loss += val_ref_rec_loss / num_val_data
                    running_avg_val_kl += val_kl / num_val_data
                    running_avg_val_inf_mse += val_inf_mse / num_val_data
                    running_avg_val_inf_ssim += val_inf_ssim / num_val_data
                    running_avg_val_inf_psnr += val_inf_psnr / num_val_data

                    val_ref_img_list.append(val_ref_img)
                    val_inf_img_list.append(val_inf_img)

                image_path = os.path.join(cf.experiment_image_dir,
                                          'epoch_{}_val_samples.png'.format(i//cf.val_every_n_batches))
                training_utils.save_sample_img(val_data_clean_list, val_ref_img_list, val_inf_img_list,
                                               img_path=image_path,
                                               noise_type=cf.noise_type, noise_param=cf.noise_param,
                                               colormap=cf.colormap)

                val_summary = sess.run(validation_summary_op,
                                       feed_dict={val_avg_ref_rec_loss: running_avg_val_ref_rec_loss,
                                                  val_avg_kl: running_avg_val_kl,
                                                  val_avg_inf_mse: running_avg_val_inf_mse,
                                                  val_avg_inf_ssim: running_avg_val_inf_ssim,
                                                  val_avg_inf_psnr: running_avg_val_inf_psnr})
                summary_writer.add_summary(val_summary, i)

                if cf.disable_progress_bar:
                    logging.info('Evaluating epoch {}/{}: validation loss={}, kl={}'
                                 .format(i, cf.num_training_batches, running_avg_val_ref_rec_loss, running_avg_val_kl))
            sess.run(global_step)

        training_time_delta = time.time() - training_start_time
        logging.info('Total training time (with running time validations) is: %f' % training_time_delta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of PRL with DnCNN')
    parser.add_argument('-c', '--config', type=str, default='./config/config_template.py',
                        help='path of the configuration file of this training')
    args = parser.parse_args()

    cf = SourceFileLoader('cf', args.config).load_module()

    if not os.path.isdir(cf.experiment_dir):
        os.mkdir(cf.experiment_dir)

    if not os.path.isdir(cf.experiment_image_dir):
        os.mkdir(cf.experiment_image_dir)

    log_path = os.path.join(cf.experiment_dir, 'train.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Logging to {}'.format(log_path))
    tf.logging.set_verbosity(tf.logging.INFO)

    train(cf)
