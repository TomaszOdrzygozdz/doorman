from keras.layers import Dense, Input, Layer
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers

import keras
import tensorflow as tf

class ConfidenceNetwork:
    def __init__(self, obs_dim, hidden_sizes):
        obs_input = Input(batch_shape=(None, obs_dim))
        network_layer = obs_input
        for layer_size in hidden_sizes:
            network_layer = Dense(layer_size, activation='relu')(network_layer)

        mlp_outputs = Dense(obs_dim)(network_layer)
        confidence = Dense(obs_dim, activation='softmax')(network_layer)
        self.model = Model(obs_input, [mlp_outputs, confidence], name="decoder")

class ConfidenceMLPModel(Model):
    def __init__(self, obs_dim, hidden_sizes, **kwargs):
        super(ConfidenceMLPModel, self).__init__(**kwargs)

        self.confidence_model = ConfidenceNetwork(obs_dim, hidden_sizes).model

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            reconstruction, confidence = self.confidence_model(x)
            #uniformization_loss = -tf.reduce_mean(tf.math.log(tf.math.scalar_mul(1/self.obs_dim, confidence)))

            reconstruction_conf = tf.math.multiply(reconstruction, confidence)
            y_conf = tf.math.multiply(y, confidence)

            reconstruction_loss_conf = tf.reduce_mean(
                keras.losses.mean_squared_error(y_conf, reconstruction_conf)
            )

            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(y, reconstruction)
            )

            reconstruction_loss *= 100

            total_loss = reconstruction_loss + reconstruction_loss_conf #+ uniformization_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "uniformization_loss_conf" : reconstruction_loss_conf
        }

class ConfidenceMLP:
    def __init__(self, obs_dim, hidden_sizes):
        self.confidence_model = ConfidenceMLPModel(obs_dim, hidden_sizes)
        self.confidence_model.compile(optimizer=Adam())

    def fit(self, x, y, epochs):
        self.confidence_model.fit(x, y, epochs=epochs)

    def predict(self, x):
        return self.confidence_model.decoder(x)


