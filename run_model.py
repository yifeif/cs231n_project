"""Runs pix2pix model based on https://arxiv.org/pdf/1611.07004.pdf."""
import argparse
from data import data_loader
import model
import os
import tensorflow as tf


def main():
  data_split_dir = FLAGS.data_split_dir
  if not data_split_dir:
    data_split_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'data')
  num_epoch = 1 # 10

  train_models_file = os.path.join(data_split_dir, 'train_data.txt')
  edges_batch, images_batch = (
      data_loader.input(FLAGS.screenshots_dir, train_models_file, batch_size=16))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    model.run_a_gan(sess, edges_batch, images_batch, num_epoch=num_epoch)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument(
      '--screenshots_dir', type=str, default=None, required=True,
      help='Path to the screenshots directory containing data images.')
  parser.add_argument(
      '--data_split_dir', type=str, default=None, required=False,
      help='Path to directory that contains test_data.txt, '
           'val_data.txt and train_data.txt files.')

  FLAGS, _ = parser.parse_known_args()
  main()
