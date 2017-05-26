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


def discriminator(x, training=True, model_size=ModelSize.MODEL_256):
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
          a1 = tf.layers.conv2d(x, 64, (4,4), strides=(2,2), kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
          # layer_2: [batch, 128, 128, 128] => [batch, 64, 64, 128]
          a2 = tf.layers.conv2d(a1, 128, (4,4), strides=(2,2), kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
          a2_bn = tf.layers.batch_normalization(a2, training=training)
        else:
          a2_bn = x
        # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256]
        a3 = tf.layers.conv2d(a2_bn, 256, (4,4), strides=(2,2), kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a3_bn = tf.layers.batch_normalization(a3, training=training)
        # layer_4: [batch, 32, 32, 256] => [batch, 31, 31, 512]
        a4 = tf.layers.conv2d(a3_bn, 512, (4,4), strides=(1,1), kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a4_bn = tf.layers.batch_normalization(a4, training=training)

        # layer_5: [batch, 32, 32, 512] => [batch, 30, 30, 512]
        a5 = tf.layers.conv2d(a4_bn, 512, (4,4), strides=(1,1), kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a5_bn = tf.layers.batch_normalization(a5, training=training)

        # layer_6: [batch, 32, 32, 512] => [batch, 29, 29, 512]
        a6 = tf.layers.conv2d(a5_bn, 512, (4,4), strides=(1,1), kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a6_bn = tf.layers.batch_normalization(a6, training=training)

        #logits = tf.layers.conv2d(a6_bn, 1, (29,29), strides=(1,1), kernel_initializer=tf.random_normal_initializer(0, 0.02))
        logits = tf.layers.conv2d(a6_bn, 1, (21,21), strides=(1,1), kernel_initializer=tf.random_normal_initializer(0, 0.02))

    return logits

       
def generator(
    d, training=True, dropout_training=True, model_size=ModelSize.MODEL_256):
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
        # Encoder:
        if model_size == ModelSize.MODEL_256:
          # layer_1: [batch, 256, 256, 1] => [batch, 128, 128, 64]
          a1 = tf.layers.conv2d(d, 64, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
          # layer_2: [batch, 128, 128, 128] => [batch, 64, 64, 128]
          a2 = tf.layers.conv2d(a1, 128, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
          a2_bn = tf.layers.batch_normalization(a2, training=training)
        else:
          a2_bn = d
        # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256]
        a3 = tf.layers.conv2d(a2_bn, 256, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a3_bn = tf.layers.batch_normalization(a3, training=training)
        # layer_4: [batch, 32, 32, 256] => [batch, 16, 16, 512]
        a4 = tf.layers.conv2d(a3_bn, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a4_bn = tf.layers.batch_normalization(a4, training=training)
        # layer_5: [batch, 16, 16, 512] => [batch, 8, 8, 512]
        a5 = tf.layers.conv2d(a4_bn, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a5_bn = tf.layers.batch_normalization(a5, training=training)
        # layer_6: [batch, 8, 8, 512] => [batch, 4, 4, 512]
        a6 = tf.layers.conv2d(a5_bn, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a6_bn = tf.layers.batch_normalization(a6, training=training)
        # layer_7: [batch, 4, 4, 512] => [batch, 2, 2, 512]
        a7 = tf.layers.conv2d(a6_bn, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a7_bn = tf.layers.batch_normalization(a7, training=training)
        # layer_8: [batch, 2, 2, 512] => [batch, 1, 1, 512]
        a8 = tf.layers.conv2d(a7_bn, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a8_bn = tf.layers.batch_normalization(a8, training=training)

        # Decoder
        d8 = tf.layers.conv2d_transpose(a8_bn, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d8_bn = tf.layers.batch_normalization(d8, training=training)
        d8_dropout = tf.layers.dropout(d8_bn, dropout_p, training=dropout_training)
        d8_unet = tf.concat([d8_dropout, a7_bn], 3)

        d7 = tf.layers.conv2d_transpose(d8_unet, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d7_bn = tf.layers.batch_normalization(d7, training=training)
        d7_dropout = tf.layers.dropout(d7_bn, dropout_p, training=dropout_training)
        d7_unet = tf.concat([d7_dropout, a6_bn], 3)

        d6 = tf.layers.conv2d_transpose(d7_unet, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d6_bn = tf.layers.batch_normalization(d6, training=training)
        d6_dropout = tf.layers.dropout(d6_bn, dropout_p, training=dropout_training)
        d6_unet = tf.concat([d6_dropout, a5_bn], 3)

        d5 = tf.layers.conv2d_transpose(d6_unet, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d5_bn = tf.layers.batch_normalization(d5, training=training)
        d5_unet = tf.concat([d5_bn, a4_bn], 3)

        d4 = tf.layers.conv2d_transpose(d5_unet, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d4_bn = tf.layers.batch_normalization(d4, training=training)
        d4_unet = tf.concat([d4_bn, a3_bn], 3)

        d3 = tf.layers.conv2d_transpose(d4_unet, 256, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d3_bn = tf.layers.batch_normalization(d3, training=training)

        if model_size == ModelSize.MODEL_256:
          d3_unet = tf.concat([d3_bn, a2_bn], 3)
          d2 = tf.layers.conv2d_transpose(d3_unet, 128, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
          d2_bn = tf.layers.batch_normalization(d2, training=training)
          d2_unet = tf.concat([d2_bn, a1], 3)

          d1 = tf.layers.conv2d_transpose(d2_unet, 64, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
          d1_bn = tf.layers.batch_normalization(d1, training=training)
        else:
          d1_bn = d3_bn

        img = tf.layers.conv2d(d1_bn, 3, (1, 1), strides=(1,1), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.tanh)

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
