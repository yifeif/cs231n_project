"""Runs pix2pix model based on https://arxiv.org/pdf/1611.07004.pdf."""
import argparse
import logging
from datetime import datetime
from data import data_loader
from model import *
import os
from scipy.misc import imsave
import tensorflow as tf


def save_images(samples, sample_images_dir, label):
  if sample_images_dir:
    iteration_dir = os.path.join(sample_images_dir, label)
    if not os.path.isdir(iteration_dir):
      os.makedirs(iteration_dir)
    for sample_index in range(samples.shape[0]):
      imsave(os.path.join(iteration_dir, 'output-%d.png' % sample_index),
             (samples[sample_index] + 1) * (128.0))


# a giant helper function
def run_a_gan(sess, data_split_dir, num_examples,
              show_every=1000, print_every=100, num_epoch=15):
  """Train a GAN for a certain number of epochs.

  Inputs:
  - sess: A tf.Session that we want to use to run our data
  - G_train_step: A training step for the Generator
  - G_loss: Generator loss
  - D_train_step: A training step for the Generator
  - D_loss: Discriminator loss
  - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
  - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
  - edges_batch: A (N, 256, 256, 1) Tensor representing batch of edge images.
  - images_batch: A (N, 256, 256, 3) Tensor representing batch of colored
    images.
  Returns:
    Nothing
  """
  global_step = tf.contrib.framework.get_or_create_global_step()
  summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
  training = tf.placeholder(tf.bool)

  train_models_file = os.path.join(data_split_dir, 'train_data.txt')
  edges_batch, images_batch = (
      data_loader.input(FLAGS.screenshots_dir, train_models_file, batch_size=4))
  batch_size = int(edges_batch.shape[0])

  val_models_file = os.path.join(data_split_dir, 'val_data.txt')
  vedges_batch, vimages_batch = (
      data_loader.input(FLAGS.screenshots_dir, val_models_file, batch_size=16))

  edges_batch_placeholder = tf.placeholder(tf.float32, (None, 256, 256, 1))
  images_batch_placeholder = tf.placeholder(tf.float32, (None, 256, 256, 3))

  sample_images_dir = os.path.join(FLAGS.sample_images_dir, "train")
  vsample_images_dir = os.path.join(FLAGS.sample_images_dir, "validation")
  # Create model
  # placeholder for images from the training dataset
  # x = tf.placeholder(tf.float32, [None, H, W, C])
  x = images_batch_placeholder
  # edge image
  # d = tf.placeholder(tf.float32, [None, H, W, 1])
  d = edges_batch_placeholder
  # generated images
  y = generator(d, training)

  with tf.variable_scope("") as scope:
      #scale images to be -1 to 1
      logits_real = discriminator(x, training)
      # Re-use discriminator weights on new inputs
      scope.reuse_variables()
      logits_fake = discriminator(y, training)

  # Get the list of variables for the discriminator and generator
  D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

  # get our solver
  D_solver, G_solver = get_solvers()

  # get our loss
  D_loss, G_loss = gan_loss(logits_real, logits_fake, x, y)
  tf.summary.scalar('D_loss', D_loss)
  tf.summary.scalar('G_loss', G_loss)

  summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
  summary_op = tf.summary.merge(summaries)

  # setup training steps
  D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
  G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
  D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
  G_train_step = G_solver.minimize(G_loss, var_list=G_vars, global_step=global_step)

  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
  if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
  else:
      print("Created model with fresh parameters.")
      sess.run(tf.global_variables_initializer())
      print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))


  sample_images_dir = os.path.join(FLAGS.sample_images_dir, "train")
  vsample_images_dir = os.path.join(FLAGS.sample_images_dir, "validation")
  #run_model(y, D_train_step, G_train_step, D_loss, G_loss, global_step,
  #          num_examples, num_epoch, 4, summary_op, sample_images_dir, is_training=True)

  #run_model(y, D_train_step, G_train_step, D_loss, G_loss, global_step,
  #          num_examples, num_epoch, 500, summary_op, sample_images_dir, is_training=True)


  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  # Run
  # compute the number of iterations we need
  max_iter = int(num_examples*num_epoch/batch_size)
  with coord.stop_on_exception():
    for it in range(max_iter):
        edges_batch_curr, images_batch_curr = sess.run([edges_batch, images_batch])
        #print('Iteration %d' % it)
        # every show often, show a sample result
        if it % show_every == 0:
          samples = sess.run(
              y, feed_dict={training: False,
                            edges_batch_placeholder: edges_batch_curr,
                            images_batch_placeholder: images_batch_curr}) #G_sample)
          save_images(samples, sample_images_dir, str(it))
          if it >= 100:
            vedges_batch_curr, vimages_batch_curr = sess.run([vedges_batch, vimages_batch])
            val_G_loss, val_D_loss, vsamples = sess.run(
                [G_loss, D_loss, y], feed_dict={training: False,
                            edges_batch_placeholder: vedges_batch_curr,
                            images_batch_placeholder: vimages_batch_curr}) #G_sample)
            save_images(vsamples, vsample_images_dir, str(it))

          #fig = show_images(samples[:16])
          #plt.show()
          print()
        # run a batch of data through the network
        # x, e = sample_from_data()
        _, D_loss_curr = sess.run([D_train_step, D_loss],
                            feed_dict={training: True,
                            edges_batch_placeholder: edges_batch_curr,
                            images_batch_placeholder: images_batch_curr}) #G_sample)
        _, G_loss_curr, gs = sess.run(
            [G_train_step, G_loss, global_step],
            feed_dict={training: True,
                       edges_batch_placeholder: edges_batch_curr,
                       images_batch_placeholder: images_batch_curr}) #G_sample)

        summary_str = sess.run(summary_op,
                         feed_dict={training: True,
                       edges_batch_placeholder: edges_batch_curr,
                       images_batch_placeholder: images_batch_curr}) #G_sample)


        summary_writer.add_summary(summary_str, gs)

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))

        # per epoch
        if it % int(np.ceil(num_examples/batch_size)) == 0:
            checkpoint_file = os.path.join(FLAGS.train_dir, "{:%Y%m%d_%H%M%S}".format(datetime.now()))
            print("Saving variables to '%s'." % checkpoint_file)
            saver.save(sess, checkpoint_file, global_step=global_step)

  coord.request_stop()
  coord.join(threads)
  print('Final images')
  samples = sess.run(y, feed_dict={training: False,
                       edges_batch_placeholder: edges_batch_curr,
                       images_batch_placeholder: images_batch_curr}) #G_sample)

  save_images(samples, sample_images_dir, str(it))

  #fig = show_images(samples[:16])
  #plt.show()

def main():
  data_split_dir = FLAGS.data_split_dir
  if not data_split_dir:
    data_split_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'data')
  num_epoch = 15

  with tf.Session() as sess:
    run_a_gan(
        sess, data_split_dir, num_examples=90000, num_epoch=num_epoch)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument(
      '--screenshots_dir', type=str, default=None, required=True,
      help='Path to the screenshots directory containing data images.')
  parser.add_argument(
      '--train_dir', type=str, default='/data/annarev/checkpoints', required=False,
      help='Path to the chkpt files.')
  parser.add_argument(
      '--data_split_dir', type=str, default=None, required=False,
      help='Path to directory that contains test_data.txt, '
           'val_data.txt and train_data.txt files.')
  parser.add_argument(
      '--sample_images_dir', type=str, default='/data/annarev/output_images', required=False,
      help='Directory to store output images at.')

  FLAGS, _ = parser.parse_known_args()
  main()
