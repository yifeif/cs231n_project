"""Utilities for loading ShapeNet screenshot data."""
import argparse
import numpy as np
import os
import tensorflow as tf


_ORIGINAL_SIZE = 256
_ORIGINAL_SIZE = 256
_CHANNELS = 3
_ORIENTATIONS_PER_MODEL = 10


def get_model_paths(model_paths_file, screenshot_dir,
                    filter_str=None):
  """Reads model paths from file.

  Args:
    model_paths_file: path to either train_data.txt, train_data.txt or
      val_data.txt generated by split_data.py.

  Returns:
    List of string model paths (relative to screenshots directory).
  """
  with open(model_paths_file, 'r') as f:
    path_suffixes = f.read().split('\n')
    if filter_str is None:
      return [os.path.join(screenshot_dir, path_suffix)
              for path_suffix in path_suffixes]
    else:
      return [os.path.join(screenshot_dir, path_suffix)
              for path_suffix in path_suffixes if filter_str in path_suffix]


def get_inputs_for_model_paths(model_paths):
  """Gets a tuples of inputs for each path in model_paths.

  Args:
    model_paths: List of string model directory paths to get inputs from.

  Returns:
    A tuple of equally-sized lists:
    List of paths to edge detection output files in .bin format.
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

  return edge_files, image_files, orientations


def get_inputs_with_orientations(
    edges_paths, image_paths, orientations, input_orientation):
  edges_paths_1 = []
  image_paths_1 = []
  orientations_1 = []
  edges_paths_2 = []
  image_paths_2 = []
  orientations_2 = []

  # By default create inputs for all orientations.
  start_orientation = 0
  step = 1
  # If input orientation is specified, only use that orientation
  # as the input.
  if input_orientation is not None:
    start_orientation = input_orientation
    step = 10

  for i in range(start_orientation, len(edges_paths), step):
    model_index = int(i/_ORIENTATIONS_PER_MODEL) * _ORIENTATIONS_PER_MODEL
    orientation_index = i - model_index
    for o in range(_ORIENTATIONS_PER_MODEL):
      edges_paths_1.append(edges_paths[i])
      image_paths_1.append(image_paths[i])
      orientations_1.append(orientation_index)
      image_paths_2.append(image_paths[model_index+o])
      edges_paths_2.append(edges_paths[model_index+o])
      orientations_2.append(orientations[model_index+o])
  return (edges_paths_1, image_paths_1, orientations_1,
          edges_paths_2, image_paths_2, orientations_2)


def input(
    screenshots_dir, model_list_file, batch_size=4, image_size=256):
  """Creates an input for ShapeNet screenshots and edge data.

  Args:
    screenshots_dir: directory to ShapeNet screenshots and edges data.
    model_list_file: Path to file that contains a list of paths
      relative to screenshots_dir for the models that we want to load
      either for training, validation or testing.
    image_size: (Integer) size of images to get. Images are square in
      size, so WIDTH == HEIGHT == image_size.

  Returns:
    A tuple of:
    - edges_batch: (N, image_size, image_size, 1) Tensor of input images (edges).
    - images_batch: (N, image_size, image_size, 3) Tensor of training output images.

  Raises:
    IOError if model_list_file is not a file.
  """
  if not os.path.isfile(model_list_file):
    raise IOError('%s does not exist.' % model_list_file)

  edges_paths, image_paths, _ = get_inputs_for_model_paths(
      get_model_paths(model_list_file, screenshots_dir))
  input_queue = tf.train.slice_input_producer([edges_paths, image_paths])

  edges = tf.decode_raw(tf.read_file(input_queue[0]), out_type=tf.float32)
  edges = tf.cast(tf.reshape(edges, [_ORIGINAL_SIZE, _ORIGINAL_SIZE, 1]), dtype=tf.float32)
  image = tf.cast(
      tf.image.decode_png(tf.read_file(input_queue[1]), channels=_CHANNELS),
      tf.float32)
  image.set_shape([_ORIGINAL_SIZE, _ORIGINAL_SIZE, 3])

  if image_size != _ORIGINAL_SIZE:
    image = tf.image.resize_images(
        image, size=[image_size, image_size], method=tf.image.ResizeMethod.AREA)
    edges = tf.image.resize_images(
        edges, size=[image_size, image_size], method=tf.image.ResizeMethod.AREA)

  # Shift and scale so that edges and image are between -1 and 1
  edges = 2*edges - 1
  image = (image / 128.0) - 1

  min_after_dequeue = 10000  # size of buffer to sample from
  num_preprocess_threads = 16
  capacity = min_after_dequeue + 3 * batch_size
  edges_batch, images_batch = tf.train.shuffle_batch(
      [edges, image], batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return edges_batch, images_batch


def multi_view_input(
    screenshots_dir, model_list_file, batch_size=4, image_size=256,
    object_type='car', input_orientation=None):
  """Creates two-view image pair input.

  Args:
    screenshots_dir: directory to ShapeNet screenshots and edges data.
    model_list_file: Path to file that contains a list of paths
      relative to screenshots_dir for the models that we want to load
      either for training, validation or testing.
    image_size: (Integer) size of images to get. Images are square in
      size, so WIDTH == HEIGHT == image_size.

  Returns:
    A tuple of:
    - edges_batch: (N, image_size, image_size, 1) Tensor of input images (edges).
    - images_batch: (N, image_size, image_size, 3) Tensor of training output images.

  Raises:
    IOError if model_list_file is not a file.
  """
  if not os.path.isfile(model_list_file):
    raise IOError('%s does not exist.' % model_list_file)

  edges_paths, image_paths, orientations = get_inputs_for_model_paths(
      get_model_paths(model_list_file, screenshots_dir,
      filter_str='/%s/' % object_type))

  (edges_paths_1, image_paths_1, orientations_1,
   edges_paths_2, image_paths_2, orientations_2
   ) = get_inputs_with_orientations(
      edges_paths, image_paths, orientations, input_orientation)

  input_queue = tf.train.slice_input_producer(
      [edges_paths_1, image_paths_1, orientations_1,
       edges_paths_2, image_paths_2, orientations_2])

  def prepare_edges(edges_file):
    edges = tf.decode_raw(tf.read_file(edges_file), out_type=tf.float32)
    edges = tf.cast(tf.reshape(edges, [_ORIGINAL_SIZE, _ORIGINAL_SIZE, 1]), dtype=tf.float32)
    if image_size != _ORIGINAL_SIZE:
      edges = tf.image.resize_images(
          edges, size=[image_size, image_size], method=tf.image.ResizeMethod.AREA)
    # Shift and scale so that edges and image are between -1 and 1
    edges = 2*edges - 1
    return edges

  def prepare_image(image_file):
    image = tf.cast(
        tf.image.decode_png(tf.read_file(image_file), channels=_CHANNELS),
        tf.float32)
    image.set_shape([_ORIGINAL_SIZE, _ORIGINAL_SIZE, 3])
    if image_size != _ORIGINAL_SIZE:
      image = tf.image.resize_images(
          image, size=[image_size, image_size], method=tf.image.ResizeMethod.AREA)
    # Shift and scale so that edges and image are between -1 and 1
    image = (image / 128.0) - 1
    return image

  def prepare_orientation(orientation):
    return tf.one_hot(orientation, 10, on_value=1., off_value=-1., dtype=tf.float32)

  edges_1 = prepare_edges(input_queue[0])
  image_1 = prepare_image(input_queue[1])
  orientation_1 = prepare_orientation(input_queue[2])
  edges_2 = prepare_edges(input_queue[3])
  image_2 = prepare_image(input_queue[4])
  orientation_2 = prepare_orientation(input_queue[5])

  min_after_dequeue = 10000  # size of buffer to sample from
  num_preprocess_threads = 16
  capacity = min_after_dequeue + 3 * batch_size
  (edges_batch_1, images_batch_1, orientation_batch_1,
   edges_batch_2, images_batch_2, orientation_batch_2) = tf.train.shuffle_batch(
      [edges_1, image_1, orientation_1, edges_2, image_2, orientation_2],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return (edges_batch_1, images_batch_1, orientation_batch_1,
          edges_batch_2, images_batch_2, orientation_batch_2)

