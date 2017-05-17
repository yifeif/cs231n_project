"""TODO(yifeif): DO NOT SUBMIT without one-line documentation for model.

TODO(yifeif): DO NOT SUBMIT without a detailed description of model.
"""
from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

H = 256
W = 256
C = 3
dropout_p = 0.5


def leaky_relu(x, alpha=0.2):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # f(x)=1(x<0)(αx)+1(x>=0)(x)
    return tf.maximum(alpha*x, x)


def discriminator(x, training=True):
    """Compute discriminator score for a batch of input images.
    256 × 256 discriminator:
    C64-C128-C256-C512-C512-C512

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 256, 256, 3]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """

    with tf.variable_scope("discriminator"):

        # layer_1: [batch, 256, 256, 3] => [batch, 128, 128, 64]
        a1 = tf.layers.conv2d(x, 64, (4,4), strides=(2,2), kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        # layer_2: [batch, 128, 128, 128] => [batch, 64, 64, 128]
        a2 = tf.layers.conv2d(a1, 128, (4,4), strides=(2,2), kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a2_bn = tf.layers.batch_normalization(a2, training=training)
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

       
def generator(d, training=True, dropout_training=True):
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
	between each layer i in the encoder and layer n−i
	in the decoder, where n is the total number of layers. The
	skip connections concatenate activations from layer i to
	layer n − i. This changes the number of channels in the
	decoder:    
    Inputs:
    - d: TensorFlow Tensor of edge image with shape [batch_size, 256, 256, 1]
  
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    # Attempting to use uninitialized value generator/batch_normalization_5/beta
    with tf.variable_scope("generator"):
        # Encoder:
        # layer_1: [batch, 256, 256, 1] => [batch, 128, 128, 64]
        a1 = tf.layers.conv2d(d, 64, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        # layer_2: [batch, 128, 128, 128] => [batch, 64, 64, 128]
        a2 = tf.layers.conv2d(a1, 128, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=leaky_relu)
        a2_bn = tf.layers.batch_normalization(a2, training=training)
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

        # Decoder
        d6 = tf.layers.conv2d_transpose(a6_bn, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d6_bn = tf.layers.batch_normalization(d6, training=training)
        d6_dropout = tf.layers.dropout(d6_bn, dropout_p, training=dropout_training)

        d5 = tf.layers.conv2d_transpose(d6_dropout, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d5_bn = tf.layers.batch_normalization(d5, training=training)
        d5_dropout = tf.layers.dropout(d5_bn, dropout_p, training=dropout_training)

        d4 = tf.layers.conv2d_transpose(d5_dropout, 512, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d4_bn = tf.layers.batch_normalization(d4, training=training)
        d4_dropout = tf.layers.dropout(d4_bn, dropout_p, training=dropout_training)

        d3 = tf.layers.conv2d_transpose(d4_dropout, 256, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d3_bn = tf.layers.batch_normalization(d3, training=training)
        d3_dropout = tf.layers.dropout(d3_bn, dropout_p, training=dropout_training)

        d2 = tf.layers.conv2d_transpose(d3_dropout, 128, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d2_bn = tf.layers.batch_normalization(d2, training=dropout_training)
        d2_dropout = tf.layers.dropout(d2_bn, dropout_p, training=dropout_training)

        d1 = tf.layers.conv2d_transpose(d2_dropout, 64, (4,4), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.relu)
        d1_bn = tf.layers.batch_normalization(d1, training=training)
        d1_dropout = tf.layers.dropout(d1_bn, dropout_p, training=dropout_training)

        img = tf.layers.conv2d(d1_dropout, 3, (1, 1), strides=(1,1), padding='same', kernel_initializer=tf.random_normal_initializer(0, 0.02), activation=tf.nn.tanh)

    return img

def gan_loss(logits_real, logits_fake, x, y):
    """Compute the GAN loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    # TODO: compute D_loss and G_loss
    D_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_real), logits_real)
    D_loss += tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_fake), logits_fake)

    G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_fake), logits_fake)
    G_loss += tf.reduce_sum(tf.abs(x - y), axis=[1,2,3])

    D_loss = tf.reduce_mean(D_loss)
    G_loss = tf.reduce_mean(G_loss)
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





# a giant helper function
def run_a_gan(sess, edges_batch, images_batch, num_examples, show_every=250, print_every=50, num_epoch=10):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    - edges_batch: A (N, 256, 256, 1) Tensor representing batch of edge images.
    - images_batch: A (N, 256, 256, 3) Tensor representing batch of colored
      images.
    Returns:
      Nothing
    """
    batch_size = int(edges_batch.shape[0])
    # Create model
    # placeholder for images from the training dataset
    # x = tf.placeholder(tf.float32, [None, H, W, C])
    x = images_batch
    # edge image
    # d = tf.placeholder(tf.float32, [None, H, W, 1])
    d = edges_batch
    # generated images
    y = generator(d)

    with tf.variable_scope("") as scope:
        #scale images to be -1 to 1
        logits_real = discriminator(x)
        # Re-use discriminator weights on new inputs
        scope.reuse_variables()
        logits_fake = discriminator(y)

    # Get the list of variables for the discriminator and generator
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

    # get our solver
    D_solver, G_solver = get_solvers()

    # get our loss
    D_loss, G_loss = gan_loss(logits_real, logits_fake, x, y)

    # setup training steps
    D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
    G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

    sess.run(tf.global_variables_initializer())
    #tf.train.start_queue_runners(sess=sess)

    # Run
    # compute the number of iterations we need
    max_iter = int(num_examples*num_epoch/batch_size)
    for it in range(max_iter):
        print('Iteration %d' % it)
        # every show often, show a sample result
        if it % show_every == 0:
          samples = sess.run(y) #G_sample)
          #fig = show_images(samples[:16])
          #plt.show()
          print()
        # run a batch of data through the network
        # x, e = sample_from_data()
        _, D_loss_curr = sess.run([D_train_step, D_loss])
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    print('Final images')
    samples = sess.run(y)

    fig = show_images(samples[:16])
    plt.show()

def main(argv=()):
  del argv  # Unused.


if __name__ == '__main__':
  app.run()
