import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Input, Layer
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers


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
        confidence = Dense(obs_dim, activation='softmax', kernel_regularizer=regularizers.l2(1e-2),
                           activity_regularizer=regularizers.l2(1e-2))(network_layer)

        self.model = Model(latent_inputs, [decoder_outputs, confidence], name="decoder")

class VEModel(keras.Model):
    """
    ## Define the VAE as a `Model` with a custom `train_step`
    """
    def __init__(self, latent_dim, hidden_sizes, obs_dim, **kwargs):
        super(VEModel, self).__init__(**kwargs)
        self.obs_dim = obs_dim
        self.encoder = Encoder(latent_dim, hidden_sizes, obs_dim).model
        self.decoder = Decoder(latent_dim, hidden_sizes, obs_dim).model

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction, confidence = self.decoder(z)

            uniformization_loss = -tf.reduce_mean(tf.math.log(tf.math.scalar_mul(1/self.obs_dim, confidence)))

            reconstruction_conf = tf.math.multiply(reconstruction, confidence)
            y_conf = tf.math.multiply(y, confidence)
            reconstruction_loss_conf = tf.reduce_mean(
                keras.losses.mean_squared_error(y_conf, reconstruction_conf)
            )

            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(y, reconstruction)
            )

            reconstruction_loss *= 100
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + reconstruction_loss_conf + kl_loss #+ uniformization_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "uniformization_loss" : uniformization_loss
        }


class VE:
    def __init__(self, latent_dim, hidden_sizes, obs_dim):
        self.ve_model = VEModel(latent_dim, hidden_sizes, obs_dim)
        self.ve_model.compile(optimizer=Adam())

    def fit(self, x, y, epochs):
        self.ve_model.fit(x, y, epochs=epochs)

    def predict(self, x):
        z_mean, z_log_var, z = self.ve_model.encoder(np.array([x]))
        #print(f'z mean = {z_mean}, z_log_var ={z_log_var}, z={z}')
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

data_1 = np.random.rand(200000,1)
data_2 = np.random.rand(200000,1)
data_3 = np.random.rand(200000,1)

data_x = np.concatenate([data_1, data_2], axis=1)
data_y = np.concatenate([data_1, data_2], axis=1)

def show_predictions(n, obs):
    print('=========================================')
    print(f' observation = {obs}')
    for _ in range(n):
        y = f.predict(obs)
        print(f' predicton = {y[0]} | conf = {y[1]}')
    print('*****************************************')

show_predictions(5, [0,0])
show_predictions(5, [0.5,0.5])

f.fit(data_x, data_y, 5)

print('After training')


show_predictions(5, [0.5,0.5])
show_predictions(5, [0.3,0.2])