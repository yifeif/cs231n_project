"""Runs pix2pix model based on https://arxiv.org/pdf/1611.07004.pdf."""
import argparse
from data import data_loader

def main():
  X_train, Y_train, _, X_val, Y_val, _, X_test, Y_test, _ = (
      data_loader.load_ShapeNet_screenshot_data(
          FLAGS.screenshots_dir, FLAGS.data_split_dir))
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
  logging.basicConfig(level=logging.DEBUG)
  main()
