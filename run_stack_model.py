"""Runs pix2pix model based on https://arxiv.org/pdf/1611.07004.pdf."""
import argparse
import logging
from datetime import datetime
from data import data_loader
from preprocess import data_utils
from model import *
import os
from scipy.misc import imsave
from skimage.color import gray2rgb
import tensorflow as tf


def run_on_test_data(
    data_split_dir, sess, y_s1_256, y_s2,
    training, edges_batch_placeholder, images_batch_placeholder):
  test_models_file = os.path.join(data_split_dir, 'test_data.txt')
  test_edges_batch, test_images_batch = data_loader.one_batch_input(
      FLAGS.screenshots_dir, test_models_file)
  test_edges_batch, test_images_batch = sess.run(
      [test_edges_batch, test_images_batch])
  y_s1_curr, y_s2_curr = sess.run(
      [y_s1_256, y_s2],
      feed_dict={training: False,
                 edges_batch_placeholder: test_edges_batch,
                 images_batch_placeholder: test_images_batch})
  # store images from y_s2 in FLAGS.test_dir
  image_count = len(test_edges_batch)
  for i in range(image_count):
    rgb_edges = np.squeeze(gray2rgb(test_edges_batch[i]))
    combined_images = np.concatenate(
        [rgb_edges, test_images_batch[i], y_s1_curr[i], y_s2_curr[i]],
        axis=1)

    image_file_path = os.path.join(FLAGS.test_dir, 'stage2-%d.png' % i)
    imsave(image_file_path, combined_images)
  print('Stored test images in %s' % FLAGS.test_dir)



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
  summary_writer = tf.summary.FileWriter(
      os.path.join(FLAGS.train_dir, 'train'), sess.graph)
  val_summary_writer = tf.summary.FileWriter(
      os.path.join(FLAGS.train_dir, 'validation'))
  # A fixed set of validation inputs to generate the progress diagram
  val_fixed_set_summary_writer = tf.summary.FileWriter(
      os.path.join(FLAGS.train_dir, 'validation_fixed_set'))
  test_summary_writer = tf.summary.FileWriter(
      os.path.join(FLAGS.train_dir, 'test'))
  training = tf.placeholder(tf.bool)

  train_models_file = os.path.join(data_split_dir, 'train_data.txt')
  edges_batch, images_batch = (
      data_loader.input(FLAGS.screenshots_dir, train_models_file, batch_size=4,
                        image_size=256))
  batch_size = int(edges_batch.shape[0])

  val_models_file = os.path.join(data_split_dir, 'val_data.txt')
  vedges_batch, vimages_batch = (
      data_loader.input(FLAGS.screenshots_dir, val_models_file, batch_size=8,
                        image_size=256))

  val_fixed_set_models_file = os.path.join(data_split_dir, 'val_fixed_set_data.txt')
  vfsedges_batch, vfsimages_batch = (
      data_loader.input(FLAGS.screenshots_dir, val_fixed_set_models_file, batch_size=20,
                        image_size=256, shuffle=False))  

  edges_batch_placeholder = tf.placeholder(tf.float32, (None, 256, 256, 1))
  images_batch_placeholder = tf.placeholder(tf.float32, (None, 256, 256, 3))



  x = images_batch_placeholder
  # edge image
  d = edges_batch_placeholder
  ############################
  # Setup stage1 model
  ############################
  if FLAGS.smaller_model:
      d_s1 = tf.image.resize_images(d, [64, 64], method=tf.image.ResizeMethod.AREA)
      stage1_model_size = ModelSize.MODEL_64 
  else:
      d_s1 = d
      stage1_model_size = ModelSize.MODEL_256 

  # generated images 
  y_s1 = generator(d_s1, training=False, decoder=FLAGS.decoder, model_size=stage1_model_size)


  #############################
  # Setup stage2 model
  #############################
  # Feed result from stage1 and concatenated with edge
  y_s1_256 = tf.image.resize_images(y_s1, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  d_s2 = tf.concat([d, y_s1_256], axis=3)

  with tf.variable_scope("s2") as scope:
      # generated images
      # No dropout in stage2
      if FLAGS.stackgan:
          print('Use stackgan architecture for stage2')
          y_s2 = stage2_generator(d_s2, training)
      else:
          print('Use same architecture as stage1 for stage2')
          y_s2 = generator(d_s2, training, dropout_training=False, decoder=FLAGS.decoder, model_size=ModelSize.MODEL_256)
      
      #TODO: condition on d or d_s2 here?
      #scale images to be -1 to 1
      logits_real = discriminator(tf.concat([d, x], axis=3), training, model_size=ModelSize.MODEL_256)
      # Re-use discriminator weights on new inputs
      scope.reuse_variables()
      logits_fake = discriminator(tf.concat([d, y_s2], axis=3), training,
                                     model_size=ModelSize.MODEL_256)

  # Get the list of variables for the discriminator and generator
  D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 's2/discriminator')
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 's2/generator')
  print(D_vars)
  print(G_vars)

  # get our solver
  D_solver, G_solver = get_solvers(learning_rate=2e-4)

  # get our loss
  D_loss, G_loss = gan_loss(
      logits_real, logits_fake, x, y_s2, lambda_param=100.0)

  with tf.variable_scope("train_summaries") as scope:
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)
  summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, "train_summaries")
  summary_op = tf.summary.merge(summaries)

  with tf.variable_scope("val_summaries") as scope:
    edges_3_channels = tf.image.grayscale_to_rgb(edges_batch_placeholder)
    tf.summary.image(
        'Images',
        tf.concat([edges_3_channels, images_batch_placeholder, y_s1_256, y_s2],
                  axis=2), max_outputs=4)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)
  val_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, "val_summaries")
  val_summary_op = tf.summary.merge(val_summaries)

  with tf.variable_scope("val_fixed_set_summaries") as scope:
    edges_3_channels = tf.image.grayscale_to_rgb(edges_batch_placeholder)
    tf.summary.image(
        'Images',
        tf.concat([edges_3_channels, images_batch_placeholder, y_s1_256, y_s2],
                  axis=2), max_outputs=20)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)
  val_fixed_set_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, "val_fixed_set_summaries")
  val_fixed_set_summary_op = tf.summary.merge(val_fixed_set_summaries)

  with tf.variable_scope("test_summaries") as scope:
    edges_3_channels = tf.image.grayscale_to_rgb(edges_batch_placeholder)
    tf.summary.image(
        'Images',
        tf.concat([edges_3_channels, y_s1_256, y_s2],
                  axis=2), max_outputs=4)
  test_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, "test_summaries")
  test_summary_op = tf.summary.merge(test_summaries)


  # setup training steps
  D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 's2/discriminator')
  G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 's2/generator')
  with tf.control_dependencies(D_extra_step):
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
  with tf.control_dependencies(G_extra_step):
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars, global_step=global_step)
  

  ###################################
  # Load ckpts for stage1 and maybe stage2
  ##################################
  saver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith('s2/')])
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
  if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
      print("Reading stage2 parameters from %s" % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
  else:
      print("Created stage2 model with fresh parameters.")
      sess.run(tf.global_variables_initializer())
      print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
  
  # Load Stage1 checkpoints
  saver_s1 = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith('generator/')])
  ckpt_s1 = tf.train.get_checkpoint_state(FLAGS.stage1_train_dir)
  v2_path_s1 = ckpt_s1.model_checkpoint_path + ".index" if ckpt_s1 else ""
  if ckpt_s1 and (tf.gfile.Exists(ckpt_s1.model_checkpoint_path) or tf.gfile.Exists(v2_path_s1)):
      print("Reading stage1 parameters from %s" % ckpt_s1.model_checkpoint_path)
      saver_s1.restore(sess, ckpt_s1.model_checkpoint_path)
  else:
      raise ValueError('Cannot load stage1 checkpoint from FLAGS.stage1_train_dir')


  if FLAGS.test_sketch:
    sketch_input = data_utils.sketch_to_edge(FLAGS.test_sketch)
    print(sketch_input.shape)
          
    test_summary_str, y_s2_curr = sess.run([test_summary_op, y_s2],
        feed_dict={training: False, edges_batch_placeholder: sketch_input})
    test_summary_writer.add_summary(test_summary_str, 1)
    print('Stored test images.')
    return

  if FLAGS.test_dir:
    run_on_test_data(
        data_split_dir, sess, y_s1_256, y_s2,
        training, edges_batch_placeholder, images_batch_placeholder)
    return


  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  # Run
  # compute the number of iterations we need
  max_iter = int(num_examples*num_epoch/batch_size)
  with coord.stop_on_exception():
    for it in range(max_iter):
        edges_batch_curr, images_batch_curr = sess.run([edges_batch, images_batch])
        # Every so often, add training and validation images to summary
        if it % show_every == 0:
            vedges_batch_curr, vimages_batch_curr = sess.run([vedges_batch, vimages_batch])
            val_summary_str, val_G_loss = sess.run(
                [val_summary_op, G_loss], feed_dict={training: False,
                                           edges_batch_placeholder: vedges_batch_curr,
                                           images_batch_placeholder: vimages_batch_curr})
            val_summary_writer.add_summary(val_summary_str, it)
            print('Validation loss %f:' % val_G_loss)
            print('Stored validation images.')
        # Every so often, add fixed validation result to summary
        if it % 2000 == 0:
            vfsedges_batch_curr, vfsimages_batch_curr = sess.run([vfsedges_batch, vfsimages_batch])
            val_fixed_set_summary_str, val_fixed_set_G_loss = sess.run(
                [val_fixed_set_summary_op, G_loss], feed_dict={training: False,
                edges_batch_placeholder: vfsedges_batch_curr,
                images_batch_placeholder: vfsimages_batch_curr})
            val_fixed_set_summary_writer.add_summary(val_fixed_set_summary_str, it)
            print('Validation loss %f:' % val_fixed_set_G_loss)
            print('Stored validation fixed set images.')

        # run a batch of data through the network
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

        # per quarter epoch
        if it % int(np.ceil(num_examples/batch_size/10)) == 0:
            checkpoint_file = os.path.join(FLAGS.train_dir, "{:%Y%m%d_%H%M%S}".format(datetime.now()))
            print("Saving variables to '%s'." % checkpoint_file)
            saver.save(sess, checkpoint_file, global_step=global_step)

  coord.request_stop()
  coord.join(threads)


def main():
  data_split_dir = FLAGS.data_split_dir
  if not data_split_dir:
    data_split_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'data')
  num_epoch = 15

  with tf.Session() as sess:
    run_a_gan(
        sess, data_split_dir, num_examples=95170, num_epoch=num_epoch)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument(
      '--screenshots_dir', type=str, default=None, required=True,
      help='Path to the screenshots directory containing data images.')
  parser.add_argument(
      '--train_dir', type=str, default='/tmp/checkpoints', required=False,
      help='Path to the chkpt files.')
  parser.add_argument(
      '--data_split_dir', type=str, default=None, required=False,
      help='Path to directory that contains test_data.txt, '
           'val_data.txt and train_data.txt files.')
  parser.add_argument(
      '--decoder', type=str, default='default', required=False,
      help='Types of decoder to use. Default to pix2pix paper. Can choose from'
           'resize_conv')
  parser.add_argument(
      '--test_sketch', type=str, default=None, required=False,
      help='Path to the test image sketch. Must be of size 256x256.')
  parser.add_argument(
      '--test_dir', type=str, default=None, required=False,
      help='If set, would run the model against a test dataset and output '
      'images to the test directory')
  parser.add_argument(
      '--stage1_train_dir', type=str, default=None, required=True,
      help='Path to the stage1 chkpt files.')
  parser.add_argument(
      '--smaller_model', type='bool', default=True, required=False,
      help='Whether to run on smaller images (64x64) or full images (256x256)') 
  parser.add_argument(
      '--stackgan', type='bool', default=True, required=False,
      help='Follow stage2 in stackgan paper') 

  FLAGS, _ = parser.parse_known_args()
  main()
