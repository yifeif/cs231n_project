"""Utilities for loading ShapeNet screenshot data."""
import argparse
from enum import Enum
import numpy as np
import os
import tensorflow as tf


_WIDTH = 256
_HEIGHT = 256
_CHANNELS = 3
_ORIENTATIONS_PER_MODEL = 10


def get_model_paths(model_paths_file, screenshot_dir):
  """Reads model paths from file.

  Args:
    model_paths_file: path to either train_data.txt, train_data.txt or
      val_data.txt generated by split_data.py.

  Returns:
    List of string model paths (relative to screenshots directory).
  """
  with open(model_paths_file, 'r') as f:
    path_suffixes = f.read().split('\n')
    return [os.path.join(screenshot_dir, path_suffix)
            for path_suffix in path_suffixes]


def get_inputs_for_model_paths(model_paths):
  """Gets a tuples of inputs for each path in model_paths.

  Args:
    model_paths: List of string model directory paths to get inputs from.

  Returns:
    A tuple of equally-sized lists:
    List of paths to edge detection output files in .npy format.
    List of paths to image files.
    List of orientations.
  """
  edge_files = []
  image_files = []
  for model_path in model_paths:
    model_name = os.path.basename(model_path)
    for orientation in range(_ORIENTATIONS_PER_MODEL):
      edges_file_path = '%s/%s-%d_padded.bin' % (model_path, model_name, orientation)
      image_file_path = '%s/%s-%d_padded.png' % (model_path, model_name, orientation)
      edge_files.append(edges_file_path)
      image_files.append(image_file_path)
  orientations = list(range(_ORIENTATIONS_PER_MODEL))*len(model_paths)
  print('Inputs for %s' % model_paths[0])
  print(str(edge_files))
  print(str(image_files))
  print(str(orientations))

  return edge_files, image_files, orientations


def input(
    screenshots_dir, model_list_file, batch_size=4):
  """Creates an input for ShapeNet screenshots and edge data.

  Args:
    screenshots_dir: directory to ShapeNet screenshots and edges data.
    model_list_file: Path to file that contains a list of paths
      relative to screenshots_dir for the models that we want to load
      either for training, validation or testing.

  Returns:
    A tuple of:
    - edges_batch: (N, 256, 256, 1) Tensor of input images (edges).
    - images_batch: (N, 256, 256, 3) Tensor of training output images.

  Raises:
    IOError if model_list_file is not a file.
  """
  if not os.path.isfile(model_list_file):
    raise IOError('%s does not exist.' % model_list_file)

  edges_paths, image_paths, _ = get_inputs_for_model_paths(
      get_model_paths(model_list_file, screenshots_dir))
  input_queue = tf.train.slice_input_producer([edges_paths, image_paths])

  edges = tf.decode_raw(tf.read_file(input_queue[0]), out_type=tf.float32)
  edges = tf.cast(tf.reshape(edges, [_HEIGHT, _WIDTH, 1]), dtype=tf.float32)
  image = tf.image.decode_png(tf.read_file(input_queue[1]), channels=_CHANNELS)
  image.set_shape([_HEIGHT, _WIDTH, 3])

  min_after_dequeue = 1 #0000  # size of buffer to sample from
  num_preprocess_threads = 1
  capacity = min_after_dequeue + 3 * batch_size
  edges_batch, images_batch = tf.train.shuffle_batch(
      [edges, image], batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=capacity,
      min_after_dequeue=min_after_dequeue)

  tf.summary.image('images', images_batch)
  return edges_batch, images_batch
