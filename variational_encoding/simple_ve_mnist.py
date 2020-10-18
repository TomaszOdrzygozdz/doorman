import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder:
    def __init__(self, latent_dim):
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.model = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

class Decoder:
    def __init__(self, latent_dim):
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        self.model = keras.Model(latent_inputs, decoder_outputs, name="decoder")

class VE(keras.Model):
    """
    ## Define the VAE as a `Model` with a custom `train_step`
    """
    def __init__(self, latent_dim, **kwargs):
        super(VE, self).__init__(**kwargs)
        self.encoder = Encoder(latent_dim=latent_dim).model
        self.decoder = Decoder(latent_dim=latent_dim).model
        self.compile(optimizer=keras.optimizers.Adam())


    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(y, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }



"""
## Train the VAE
"""
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)

mnist_digits_rotated = []
for digit in mnist_digits:
    mnist_digits_rotated.append(np.rot90(digit,1))
mnist_digits_rotated = np.array(mnist_digits_rotated)

mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
mnist_digits_rotated = np.expand_dims(mnist_digits_rotated, -1).astype("float32") / 255

ve = VE(2)

ve.fit(x=mnist_digits, y=mnist_digits_rotated, epochs=10, batch_size=128)


import matplotlib.pyplot as plt

#
# def plot_latent(encoder, decoder):
#     # display a n*n 2D manifold of digits
#     n = 30
#     digit_size = 28
#     scale = 2.0
#     figsize = 15
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-scale, scale, n)
#     grid_y = np.linspace(-scale, scale, n)[::-1]
#
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[
#                 i * digit_size : (i + 1) * digit_size,
#                 j * digit_size : (j + 1) * digit_size,
#             ] = digit
#
#     plt.figure(figsize=(figsize, figsize))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap="Greys_r")
#     plt.show()
#
# plot_latent(ve.encoder, ve.decoder)
