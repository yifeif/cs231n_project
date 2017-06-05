"""TODO(yifeif): DO NOT SUBMIT without one-line documentation for model.

TODO(yifeif): DO NOT SUBMIT without a detailed description of model.
"""
from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

dropout_p = 0.5

class ModelSize:
  MODEL_256 = 0
  MODEL_64 = 1


def leaky_relu(x, alpha=0.2):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    return tf.maximum(alpha*x, x)


def conv2d(inputs, filters, kernel_size=(4, 4), strides=(2, 2),
           padding='valid', activation=tf.nn.relu):
  return tf.layers.conv2d(
      inputs, filters, kernel_size, strides=strides, padding=padding,
      kernel_initializer=tf.random_normal_initializer(0, 0.02),
      activation=activation)

def conv2d_transpose(
    inputs, filters, kernel_size=(4, 4), strides=(2, 2), padding='same'):
  return tf.layers.conv2d_transpose(
      inputs, filters, kernel_size, strides=strides,
      padding=padding,
      kernel_initializer=tf.random_normal_initializer(0, 0.02),
      activation=tf.nn.relu)


def batch_norm(inputs, training):
  return tf.layers.batch_normalization(inputs, epsilon=1e-5, training=training)


def discriminator(x, training=True, model_size=ModelSize.MODEL_256,
                  orientations=None):
    """Compute discriminator score for a batch of input images.
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape is either
      [batch_size, 256, 256, 3] or [batch_size, 64, 64, 3] depending
      on value of "small_size".
    - small_size: If true, apply to 64x64 images instead of 256x256.
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """

    with tf.variable_scope("discriminator"):

        if model_size == ModelSize.MODEL_256:
          # layer_1: [batch, 256, 256, 3] => [batch, 128, 128, 64]
          a1 = conv2d(x, 64, activation=leaky_relu)
          # layer_2: [batch, 128, 128, 128] => [batch, 64, 64, 128]
          a2 = conv2d(a1, 128, activation=leaky_relu)
          a2_bn = batch_norm(a2, training=training)
        else:
          a2_bn = x
        # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256]
        a3 = conv2d(a2_bn, 256, activation=leaky_relu)
        a3_bn = batch_norm(a3, training=training)
        # layer_4: [batch, 32, 32, 256] => [batch, 31, 31, 512]
        a4 = conv2d(a3_bn, 512, strides=(1,1), activation=leaky_relu)
        a4_bn = batch_norm(a4, training=training)

        # layer_5: [batch, 32, 32, 512] => [batch, 30, 30, 512]
        a5 = conv2d(a4_bn, 512, strides=(1,1), activation=leaky_relu)
        a5_bn = batch_norm(a5, training=training)

        # layer_6: [batch, 32, 32, 512] => [batch, 29, 29, 512]
        a6 = conv2d(a5_bn, 512, strides=(1,1), activation=leaky_relu)
        a6_bn = batch_norm(a6, training=training)

        if orientations is not None:
          a6_bn_with_orientations = tf.concat(
              [tf.contrib.layers.flatten(a5_bn), orientations], axis=1)
          logits = tf.layers.dense(a6_bn_with_orientations, 1, activation=leaky_relu)
        else:

          logits = conv2d(
              a6_bn, 1, kernel_size=(21,21), strides=(1,1), activation=leaky_relu)
    return logits

       
def generator(
    d, training=True, dropout_training=True, decoder='default',
    model_size=ModelSize.MODEL_256, orientations=None,
    latent_vector_size=512):
    """Generate images from a random noise vector.

	The encoder-decoder architecture consists of:
	encoder:
	C64-C128-C256-C512-C512-C512-C512-C512
	decoder:
	CD512-CD512-CD512-C512-C512-C256-C128-C64
	After the last layer in the decoder, a convolution is applied
	to map to the number of output channels (3 in general,
	except in colorization, where it is 2), followed by a Tanh
	function. As an exception to the above notation, BatchNorm
	is not applied to the first C64 layer in the encoder.
	All ReLUs in the encoder are leaky, with slope 0.2, while
	ReLUs in the decoder are not leaky.
	The U-Net architecture is identical except with skip connections
        between each layer i in the encoder and layer n-i
	in the decoder, where n is the total number of layers. The
	skip connections concatenate activations from layer i to
	layer n - i. This changes the number of channels in the
	decoder:    
    Inputs:
    - d: TensorFlow Tensor of edge image with shape [batch_size, 256, 256, 1]
         or [batch_size, 256, 256, 1] depending on value of small_size.
  
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    # Attempting to use uninitialized value generator/batch_normalization_5/beta
    with tf.variable_scope("generator"):
        encoder_outputs = encoder(d, training, model_size,
                                  latent_vector_size=latent_vector_size)
        if orientations is not None:
          orientations = tf.reshape(orientations, [-1, 1, 1, 20])
          encoder_outputs['final'] = tf.concat(
              [encoder_outputs['final'], orientations], axis=3)
          return multi_view_decoder(
              encoder_outputs, training, dropout_training, model_size)

        if decoder == 'default':
          return default_decoder(
              encoder_outputs, training, dropout_training, model_size)
        elif decoder == 'resize_conv':
          return resize_conv_decoder(
              encoder_outputs, training, dropout_training, model_size)
        else:
          raise ValueError('Invalid decoder type %s' % decoder)


def stage2_generator(d, training=True):
    with tf.variable_scope("generator"):
        # 64x4x4 --> 16x16x512
        encoder_outputs = stage2_encoder(d, training)
        # 4-stage deep residual blocks
        res1 = stage2_residual_block(encoder_outputs)
        res2 = stage2_residual_block(res1)
        res3 = stage2_residual_block(res2)
        res4 = stage2_residual_block(res3)
        # 16x16x512 --> 256X256X3
        return stage2_decoder(res4, training)



def encoder(edges, training=True, model_size=ModelSize.MODEL_256,
            latent_vector_size=512):
    encoder_outputs = {}

    # Encoder:
    # layer_1: [batch, 256, 256, 1] => [batch, 128, 128, 64]
    # or [batch, 64, 64, 1] => [batch, 32, 32, 64]
    a1 = conv2d(edges, 64, padding='same', activation=leaky_relu)
    if model_size == ModelSize.MODEL_256:
      # layer_2: [batch, 128, 128, 128] => [batch, 64, 64, 128]
      a2 = conv2d(a1, 128, padding='same', activation=None)
      a2_bn = batch_norm(a2, training=training)
      a2_bn = leaky_relu(a2_bn)
      # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256]
      a3 = conv2d(a2_bn, 256, padding='same', activation=None)
      a3_bn = batch_norm(a3, training=training)
      a3_bn = leaky_relu(a3_bn)
      encoder_outputs = {'a2_bn': a2_bn, 'a3_bn': a3_bn}
    else:
      a3_bn = a1
    # layer_4: [batch, 32, 32, 256] => [batch, 16, 16, 512]
    a4 = conv2d(a3_bn, 512, padding='same', activation=None)
    a4_bn = batch_norm(a4, training=training)
    a4_bn = leaky_relu(a4_bn)
    # layer_5: [batch, 16, 16, 512] => [batch, 8, 8, 512]
    a5 = conv2d(a4_bn, 512, padding='same', activation=None)
    a5_bn = batch_norm(a5, training=training)
    a5_bn = leaky_relu(a5_bn)
    # layer_6: [batch, 8, 8, 512] => [batch, 4, 4, 512]
    a6 = conv2d(a5_bn, 512, padding='same', activation=None)
    a6_bn = batch_norm(a6, training=training)
    a6_bn = leaky_relu(a6_bn)
    # layer_7: [batch, 4, 4, 512] => [batch, 2, 2, 512]
    a7 = conv2d(a6_bn, 512, padding='same', activation=None)
    a7_bn = batch_norm(a7, training=training)
    a7_bn = leaky_relu(a7_bn)
    # layer_8: [batch, 2, 2, 512] => [batch, 1, 1, 512]
    a8 = conv2d(a7_bn, latent_vector_size, padding='same', activation=None)
    a8_bn = batch_norm(a8, training=training)
    a8_bn = leaky_relu(a8_bn)

    encoder_outputs.update({
        'a1': a1, 'a4_bn': a4_bn, 'a5_bn': a5_bn, 'a6_bn': a6_bn,
        'a7_bn': a7_bn, 'final': a8_bn})
    return encoder_outputs


def stage2_residual_block(inputs, training=True):
    a1 = conv2d(inputs, 512, kernel_size=(3, 3), strides=(1, 1), padding='same')
    a1_bn = batch_norm(a1, training=training)
    a1_relu = tf.nn.relu(a1_bn)
    a2 = conv2d(a1_relu, 512, kernel_size=(3, 3), strides=(1, 1), padding='same')
    a2_bn = batch_norm(a2, training=training)
    outputs = inputs + a2_bn
    outputs = tf.nn.relu(outputs)
    return outputs


def stage2_encoder(edges, training=True):
    # layer_1: [batch, 64, 64, 1] => [batch, 64, 64, 128]
    a1 = conv2d(edges, 128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)
    # layer_2: [batch, 64, 64, 128] => [batch, 32, 32, 256]
    a2 = conv2d(a1, 256, padding='same')
    a2_bn = batch_norm(a2, training=training)
    a2_relu = tf.nn.relu(a2_bn)
    # layer_3: [batch, 32, 32, 256] => [batch, 16, 16, 521]
    a3 = conv2d(a2_relu, 512, padding='same')
    a3_bn = batch_norm(a3, training=training)
    return tf.nn.relu(a3_bn)


def stage2_decoder(inputs, training=True):
    # -->s2 * s2 * gf_dim*2
    a1_resize = tf.image.resize_images(inputs, [32, 32], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    a1 = conv2d(a1_resize, 256, kernel_size=(3, 3), strides=(1, 1), padding='same')
    a1_relu = tf.nn.relu(a1)
    # -->s * s * gf_dim
    a2_resize = tf.image.resize_images(a1_relu, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    a2 = conv2d(a2_resize, 128, kernel_size=(3, 3), strides=(1, 1), padding='same')
    a2_relu = tf.nn.relu(a2)    
    # -->2s * 2s * gf_dim/2
    a3_resize = tf.image.resize_images(a2_relu, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    a3 = conv2d(a3_resize, 64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    a3_relu = tf.nn.relu(a3)  
    # -->4s * 4s * gf_dim//4
    a4_resize = tf.image.resize_images(a3_relu, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    a4 = conv2d(a4_resize, 32, kernel_size=(3, 3), strides=(1, 1), padding='same')
    a4_relu = tf.nn.relu(a4) 
    # -->4s * 4s * 3
    a5 = conv2d(a4_relu, 3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.tanh)
    return a5

def default_decoder(
    encoder_outputs, training=True, dropout_training=True,
    model_size=ModelSize.MODEL_256):

    d8 = conv2d_transpose(encoder_outputs['final'], 512)
    d8_bn = batch_norm(d8, training=training)
    d8_dropout = tf.layers.dropout(d8_bn, dropout_p, training=dropout_training)
    d8_unet = tf.concat([d8_dropout, encoder_outputs['a7_bn']], 3)

    d7 = conv2d_transpose(d8_unet, 512)
    d7_bn = batch_norm(d7, training=training)
    d7_dropout = tf.layers.dropout(d7_bn, dropout_p, training=dropout_training)
    d7_unet = tf.concat([d7_dropout, encoder_outputs['a6_bn']], 3)

    d6 = conv2d_transpose(d7_unet, 512)
    d6_bn = batch_norm(d6, training=training)
    d6_dropout = tf.layers.dropout(d6_bn, dropout_p, training=dropout_training)
    d6_unet = tf.concat([d6_dropout, encoder_outputs['a5_bn']], 3)

    d5 = conv2d_transpose(d6_unet, 512)
    d5_bn = batch_norm(d5, training=training)
    d5_unet = tf.concat([d5_bn, encoder_outputs['a4_bn']], 3)

    d4 = conv2d_transpose(d5_unet, 512)
    d4_bn = batch_norm(d4, training=training)

    if model_size == ModelSize.MODEL_256:
      d4_unet = tf.concat([d4_bn, encoder_outputs['a3_bn']], 3)
      d3 = conv2d_transpose(d4_unet, 256)
      d3_bn = batch_norm(d3, training=training)
      d3_unet = tf.concat([d3_bn, encoder_outputs['a2_bn']], 3)

      d2 = conv2d_transpose(d3_unet, 128)
      d2_bn = batch_norm(d2, training=training)
      d2_unet = tf.concat([d2_bn, encoder_outputs['a1']], 3)
    else:
      d2_bn = d4_bn

    d2_unet = tf.concat([d2_bn, encoder_outputs['a1']], 3)
    d1 = conv2d_transpose(d2_unet, 64)
    d1_bn = batch_norm(d1, training=training)

    img = conv2d(d1_bn, 3, kernel_size=(1, 1), strides=(1,1), padding='same', activation=tf.nn.tanh)
    return img


def resize_conv_decoder(
    encoder_outputs, training=True, dropout_training=True,
    model_size=ModelSize.MODEL_256):

    a8_bn_resize = tf.image.resize_images(encoder_outputs['final'], [2, 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d8 = conv2d(a8_bn_resize, 512, strides=(1,1), padding='same', activation=None)
    d8_bn = batch_norm(d8, training=training)
    d8_relu = tf.nn.relu(d8_bn)
    d8_dropout = tf.layers.dropout(d8_relu, dropout_p, training=dropout_training)
    d8_unet = tf.concat([d8_dropout, encoder_outputs['a7_bn']], 3)

    d8_unet_resize = tf.image.resize_images(d8_unet, [4, 4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d7 = conv2d(d8_unet_resize, 512, strides=(1,1), padding='same', activation=None)
    d7_bn = batch_norm(d7, training=training)
    d7_relu = tf.nn.relu(d7_bn)
    d7_dropout = tf.layers.dropout(d7_relu, dropout_p, training=dropout_training)
    d7_unet = tf.concat([d7_dropout, encoder_outputs['a6_bn']], 3)

    d7_unet_resize = tf.image.resize_images(d7_unet, [8, 8], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d6 = conv2d(d7_unet_resize, 512, strides=(1,1), padding='same', activation=None)
    d6_bn = batch_norm(d6, training=training)
    d6_relu = tf.nn.relu(d6_bn)
    d6_dropout = tf.layers.dropout(d6_relu, dropout_p, training=dropout_training)
    d6_unet = tf.concat([d6_dropout, encoder_outputs['a5_bn']], 3)

    d6_unet_resize = tf.image.resize_images(d6_unet, [16, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d5 = conv2d(d6_unet_resize, 512, strides=(1,1), padding='same', activation=None)
    d5_bn = batch_norm(d5, training=training)
    d5_relu = tf.nn.relu(d5_bn)
    d5_unet = tf.concat([d5_relu, encoder_outputs['a4_bn']], 3)

    d5_unet_resize = tf.image.resize_images(d5_unet, [32, 32], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d4 = conv2d(d5_unet_resize, 512, strides=(1,1), padding='same', activation=None)
    d4_bn = batch_norm(d4, training=training)
    d4_relu = tf.nn.relu(d4_bn)

    if model_size == ModelSize.MODEL_256:
      d4_unet = tf.concat([d4_relu, encoder_outputs['a3_bn']], 3)
      d4_unet_resize = tf.image.resize_images(d4_unet, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      d3 = conv2d(d4_unet_resize, 256, strides=(1,1), padding='same', activation=None)
      d3_bn = batch_norm(d3, training=training)
      d3_relu = tf.nn.relu(d3_bn)
      d3_unet = tf.concat([d3_relu, encoder_outputs['a2_bn']], 3)

      d3_unet_resize = tf.image.resize_images(d3_unet, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      d2 = conv2d(d3_unet_resize, 128, strides=(1,1), padding='same', activation=None)
      d2_bn = batch_norm(d2, training=training)
      d2_relu = tf.nn.relu(d2_bn)
      d2_unet = tf.concat([d2_relu, encoder_outputs['a1']], 3)
      d2_unet_resize = tf.image.resize_images(d2_unet, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    else:
      d2_unet = tf.concat([d4_relu, encoder_outputs['a1']], 3)
      d2_unet_resize = tf.image.resize_images(d2_unet, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    d1 = conv2d(d2_unet_resize, 64, strides=(1,1), padding='same', activation=None)
    d1_bn = batch_norm(d1, training=training)
    d1_relu = tf.nn.relu(d1_bn)

    img = conv2d(d1_relu, 3, kernel_size=(1, 1), strides=(1,1), padding='same', activation=tf.nn.tanh)

    return img


def multi_view_decoder(
    encoder_outputs, training=True, dropout_training=True,
    model_size=ModelSize.MODEL_256):

    a8_bn_resize = tf.image.resize_images(encoder_outputs['final'], [2, 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d8 = conv2d(a8_bn_resize, 512, strides=(1,1), padding='same', activation=None)
    d8_bn = batch_norm(d8, training=training)
    d8_relu = tf.nn.relu(d8_bn)
    d8_dropout = tf.layers.dropout(d8_relu, dropout_p, training=dropout_training)

    d8_resize = tf.image.resize_images(d8_dropout, [4, 4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d7 = conv2d(d8_resize, 512, strides=(1,1), padding='same', activation=None)
    d7_bn = batch_norm(d7, training=training)
    d7_relu = tf.nn.relu(d7_bn)
    d7_dropout = tf.layers.dropout(d7_relu, dropout_p, training=dropout_training)

    d7_resize = tf.image.resize_images(d7_dropout, [8, 8], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d6 = conv2d(d7_resize, 512, strides=(1,1), padding='same', activation=None)
    d6_bn = batch_norm(d6, training=training)
    d6_relu = tf.nn.relu(d6_bn)
    d6_dropout = tf.layers.dropout(d6_relu, dropout_p, training=dropout_training)

    d6_resize = tf.image.resize_images(d6_dropout, [16, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d5 = conv2d(d6_resize, 512, strides=(1,1), padding='same', activation=None)
    d5_bn = batch_norm(d5, training=training)
    d5_relu = tf.nn.relu(d5_bn)

    d5_resize = tf.image.resize_images(d5_relu, [32, 32], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d4 = conv2d(d5_resize, 512, strides=(1,1), padding='same', activation=None)
    d4_bn = batch_norm(d4, training=training)
    d4_bn = tf.nn.relu(d4_bn)

    
    d2_resize = tf.image.resize_images(d4_bn, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    d1 = conv2d(d2_resize, 64, strides=(1,1), padding='same', activation=None)
    d1_bn = batch_norm(d1, training=training)
    d1_relu = tf.nn.relu(d1_bn)

    img = conv2d(d1_relu, 3, kernel_size=(1, 1), strides=(1,1), padding='same', activation=tf.nn.tanh)

    return img



def gan_loss(logits_real, logits_fake, x, y, lambda_param):
    """Compute the GAN loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    - lambda_param: multiplier for L1 loss

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    D_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_real), logits_real)
    D_loss += tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_fake), logits_fake)

    G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_fake), logits_fake)
    G_loss += lambda_param*tf.reduce_mean(tf.abs(x - y))
    return D_loss, G_loss


def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.

    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)

    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1 = beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1 = beta1)

    return D_solver, G_solver





def main(argv=()):
  del argv  # Unused.


if __name__ == '__main__':
  app.run()
