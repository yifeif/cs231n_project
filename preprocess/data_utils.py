import os
import numpy as np
from PIL import Image
from skimage import feature
from scipy.misc import imread, imsave


# Image height before preprocessing
_ORIGINAL_HEIGHT = 192


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
  if extension == 'bin':
    #image = np.load(path)
    image = np.fromfile(path, dtype=np.bool)
    image = image.reshape((_ORIGINAL_HEIGHT, 256))
  else:
    image = imread(path)

  # Create raw padded output
  padded_image = np.empty(output_dim)
  padded_image.fill(pad_value)
  t = type(pad_value)
  padded_image = padded_image.astype(t)

  # Insert input properly
  input_dim = image.shape
  offset = np.floor((np.array(output_dim) - np.array(input_dim))/2).astype(int)
  input_index = [slice(offset[dim], offset[dim] + input_dim[dim]) for dim in range(len(input_dim))]
  padded_image[input_index] = image

  # Save output
  new_name = name + '_padded' + '.' + extension
  if extension == 'bin':
    padded_image.tofile(new_name)
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
  edge_path = rgb_image_path.rstrip('.png') + '.bin'

  edge.tofile(edge_path)
  return edge_path


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
  print('Total images number: %d' % len(images))
  for idx, image_path in enumerate(images):
    if idx % 1000 == 999:
      print('Processed %d out of %d images' % (idx+1, len(images)))
    preprocess_rawrgb(image_path)


