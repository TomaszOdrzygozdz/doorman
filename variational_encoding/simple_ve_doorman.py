import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Input, Layer
from keras.models import Model
from keras.optimizers import Adam


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder:
    def __init__(self, latent_dim, hidden_sizes, obs_dim):
        encoder_inputs = Input(batch_shape=(None, obs_dim))

        network_layer = encoder_inputs
        for layer_size in hidden_sizes:
            network_layer = Dense(layer_size, activation='relu')(network_layer)

        z_mean = Dense(latent_dim, name="z_mean")(network_layer)
        z_log_var = Dense(latent_dim, name="z_log_var")(network_layer)
        z = Sampling()([z_mean, z_log_var])
        self.model = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

class Decoder:
    def __init__(self, latent_dim, hidden_sizes, obs_dim):
        latent_inputs = Input(shape=(latent_dim,))

        network_layer = latent_inputs
        for layer_size in hidden_sizes:
            network_layer = Dense(layer_size, activation='relu')(network_layer)

        decoder_outputs = Dense(obs_dim)(network_layer)
        self.model = Model(latent_inputs, decoder_outputs, name="decoder")

class VEModel(keras.Model):
    """
    ## Define the VAE as a `Model` with a custom `train_step`
    """
    def __init__(self, latent_dim, hidden_sizes, obs_dim, **kwargs):
        super(VEModel, self).__init__(**kwargs)
        self.encoder = Encoder(latent_dim, hidden_sizes, obs_dim).model
        self.decoder = Decoder(latent_dim, hidden_sizes, obs_dim).model

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(y, reconstruction)
            )
            reconstruction_loss *= 100
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


class VE:
    def __init__(self, latent_dim, hidden_sizes, obs_dim):
        self.ve_model = VEModel(latent_dim, hidden_sizes, obs_dim)
        self.ve_model.compile(optimizer=Adam())

    def fit(self, x, y, epochs):
        self.ve_model.fit(x, y, epochs=epochs)

    def predict(self, x):
        z_mean, z_log_var, z = self.ve_model.encoder(np.array([x]))
        print(f'z mean = {z_mean}, z_log_var ={z_log_var}, z={z}')
        return self.ve_model.decoder(z)



f = VE(2,[50,50],2)
print('before training 0,0')
for _ in range(5):
    y = f.predict([0,0])
    print(y)

print('before training 0.5,0.5')
for _ in range(5):
    y = f.predict([0,0])
    print(y)

data_1 = np.random.rand(100000,1)
data_2 = np.random.rand(100000,1)
data_3 = np.random.rand(100000,1)

data_x = np.concatenate([data_1, data_2], axis=1)
data_y = np.concatenate([data_1, data_3], axis=1)
print(data_x.shape)


f.fit(data_x, data_y, 10)

print('***************************')

print('after training [0,0]')
for _ in range(5):
    y = f.predict([0,0])
    print(y)

print('after training [0.5, 0.5]')
for _ in range(5):
    y = f.predict([0.5,0.5])
    print(y)
# f.build((None,2))
# print(f.summary())
#
#
# y = f.predict_next([[0,0,0,0]])
# print(y)

"""
## Train the VAE
"""
def train():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)

    mnist_digits_rotated = []
    for digit in mnist_digits:
        mnist_digits_rotated.append(np.rot90(digit,1))
    mnist_digits_rotated = np.array(mnist_digits_rotated)

    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_digits_rotated = np.expand_dims(mnist_digits_rotated, -1).astype("float32") / 255

    ve = VE(2)

    ve.fit(x=mnist_digits, y=mnist_digits_rotated, epochs=2, batch_size=128)

    mnist_digits[0]

    import matplotlib.pyplot as plt


    def plot_latent(encoder, decoder):
        # display a n*n 2D manifold of digits
        n = 30
        digit_size = 28
        scale = 2.0
        figsize = 15
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.show()

    plot_latent(ve.encoder, ve.decoder)
