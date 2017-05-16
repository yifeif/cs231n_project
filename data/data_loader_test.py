"""Tests for data_loader."""
import data_loader
import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
import unittest
from scipy.misc import imsave

class DataLoaderTest(unittest.TestCase):

  def setUp(self):
    data_loader._ORIENTATIONS_PER_MODEL = 1
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)

  def test_load_simple_data(self):
    image = np.random.choice(range(256), size=(256, 256, 3)).astype(np.float32)
    edges = np.random.choice([True, False], size=(256, 256)).astype(np.float32)

    model_dir = os.path.join(self.test_dir, 'abc')
    os.makedirs(model_dir)
    edges.tofile(os.path.join(model_dir, 'abc-0_padded.bin'))
    imsave(os.path.join(model_dir, 'abc-0_padded.png'), image)

    model_list_path = os.path.join(self.test_dir, 'model_list.txt')
    with open(model_list_path, 'w') as model_list_file:
      model_list_file.write(model_dir)

    edges_batch, images_batch = data_loader.input(
          screenshots_dir='/', model_list_file=model_list_path, batch_size=1)
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
      sess.run(init_op)
      tf.train.start_queue_runners(sess=sess)
      edges_batch_result, images_batch_result = sess.run(
          [edges_batch, images_batch])
      self.assertEqual((1, 256, 256, 1), edges_batch_result.shape)
      self.assertEqual((1, 256, 256, 3), images_batch_result.shape)
      self.assertTrue(np.allclose(edges.reshape((256, 256, 1)),
                                  edges_batch_result[0]))
      self.assertTrue(np.allclose(image, images_batch_result[0]))


if __name__ == '__main__':
  unittest.main()

