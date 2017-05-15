import os
import numpy as np
from PIL import Image
from skimage import feature
from scipy.misc import imread, imsave


def pad_and_save_image(path, output_dim, pad_value):
  """Load image from path, center padded it to output_dim with the given
  pad_value.

  Inputs:
    path: path of input image.
    output_dim: tuple of output dimension.
    pad_value: the value to pad with.

  Returns:
    The path of the saved padded image.
  """
  name, extension = path.split('.')
  # Load input image
  if extension == 'npy':
    image = np.load(path)
  else:
    image = imread(path)

  # Create raw padded output
  padded_image = np.empty(output_dim)
  padded_image.fill(pad_value)
  t = type(pad_value)
  padded_image = padded_image.astype(t)

  # Insert input properly
  input_dim = image.shape
  offset = np.floor((np.array(output_dim) - np.array(input_dim))/2)
  input_index = [slice(offset[dim], offset[dim] + input_dim[dim]) for dim in range(len(input_dim))]
  padded_image[input_index] = image

  # Save output
  new_name = name + '_padded' + '.' + extension
  if extension == 'npy':
    np.save(new_name, padded_image)
  else:
    imsave(new_name, padded_image)
  return new_name

def save_edge(rgb_image_path, sigma=1.2):
  """Convert RGB image to edge.
  Load rgb image, convert to boolean map with edge as True,
  and save as npy file with the same name in the same dir.

  Inputs:
    RGB image path

  Returns:
    path of saved image
  """
  image = Image.open(rgb_image_path).convert('L')
  image = np.array(image)
  edge = feature.canny(image, sigma=sigma)
  edge_path = rgb_image_path.rstrip('.png')
  np.save(edge_path, edge)
  return edge_path + '.npy'


def preprocess_rawrgb(path):
  """Create and save padded input, edge, and padded edge for given input image.

  Inputs:
    path: raw rgb image path
  """
  edge_path = save_edge(path)
  pad_and_save_image(path, (256, 256, 3), np.uint8(255))
  pad_and_save_image(edge_path, (256, 256), False)


def preprocess_rawrgb_dir(path):
  """Create and save padded input, edge, and padded edge for all .png under path.

  Inputs:
    path: raw rgb image path
  """
  images = f = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.png']
  for image_path in images:
    preprocess_rawrgb(image_path)


