"""Splits data into training, validation and test datasets."""
import argparse
import glob
import logging
import os
import random


_SEED = 231
_VALIDATION_FRACTION = 0.2  # Fraction of data to keep for validation
_TEST_FRACTION = 0.1  # Fraction of data to keep for test


def split_datasets(screenshots_dir):
  """Creates train_data.txt, val_data.txt and test_data.txt files.

  These files will contain paths in the form
  screenshots/modelsByCategory/car/3dw/<model_id_dir>

  Args:
    screenshots_dir: path to the screenshots directory containing
      ShapeNet screenshots.
  """
  # Data is under
  # screenshots/modelsByCategory/car/3dw/<model_id_dir>/view-<index>.png
  base_dir = os.path.join(
      screenshots_dir, 'modelsByCategory/*/3dw/*')
  all_models = [
      model_path[len(screenshots_dir)+1:] for model_path in glob.glob(base_dir)]
  # Split all_models into training, validation and test data.
  random.seed(_SEED)
  random.shuffle(all_models)
  num_test = int(_TEST_FRACTION*len(all_models))
  num_validation = int(_VALIDATION_FRACTION*len(all_models))

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  test_models = all_models[:num_test]
  with open(os.path.join(FLAGS.output_dir, 'test_data.txt'), 'w') as f:
    f.write('\n'.join(test_models))

  validation_models = all_models[num_test:num_test+num_validation]
  with open(os.path.join(FLAGS.output_dir, 'val_data.txt'), 'w') as f:
    f.write('\n'.join(validation_models))

  train_models = all_models[num_test+num_validation:]
  with open(os.path.join(FLAGS.output_dir, 'train_data.txt'), 'w') as f:
    f.write('\n'.join(train_models))
  logging.info(
      'test_data.txt, val_data.txt and train_data.txt written to %s.' % FLAGS.output_dir)


def main():
  split_datasets(FLAGS.screenshots_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument(
      '--screenshots_dir', type=str, default=None, required=True,
      help='Path to the screenshots directory containing data images.')
  parser.add_argument(
      '--output_dir', type=str, default=None, required=True,
      help='Directory to store files containing training/validation/test datasets.')

  FLAGS, _ = parser.parse_known_args()
  logging.basicConfig(level=logging.DEBUG)
  main()
