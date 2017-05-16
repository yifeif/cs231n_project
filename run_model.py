"""Runs pix2pix model based on https://arxiv.org/pdf/1611.07004.pdf."""
import argparse
from data import data_loader
import os

def main():
  train_models_file = os.path.join(FLAGS.data_split_dir, 'train_data.txt')
  edges_batch, images_batch = (
      data_loader.input(FLAGS.screenshots_dir, train_models_file))
  # TODO: setup model that uses training data above.
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument(
      '--screenshots_dir', type=str, default=None, required=True,
      help='Path to the screenshots directory containing data images.')
  parser.add_argument(
      '--data_split_dir', type=str, default=None, required=True,
      help='Path to directory that contains test_data.txt, '
           'val_data.txt and train_data.txt files.')

  FLAGS, _ = parser.parse_known_args()
  main()
