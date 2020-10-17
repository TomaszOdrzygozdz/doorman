from typing import List
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import cosine_similarity


class GoalNetwork:

    def __init__(self, n_keys, hidden_layers):

        input = Input(batch_shape=(None, 2 + n_keys))
        layer = input

        for num, hidden_size in enumerate(hidden_layers):
            layer = Dense(hidden_size, activation='relu')(layer)

        output = Dense(2 + n_keys)(layer)

        self.model = Model(inputs=input, outputs=output)
        self.model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy', 'mean_squared_error'])

    def predict_goal(self, obs):
        return self.model.predict([obs])

class GoalNetworkEnsemble:

    def __init__(self, n_keys, hidden_layers, n_networks):
        self.n_keys = n_keys
        self.hidden_layers = hidden_layers
        self.n_networks = n_networks
        self.ensemble = []
        for _ in range(self.n_networks):
            self.ensemble.append(GoalNetwork(self.n_keys, self.hidden_layers))


    def fit_trajectories(self, trajectories, shift, epochs, mode='only_success'):

        x = []
        y = []

        used_trajectories = 0
        for trajectory in trajectories.values():

            if mode == 'all' or (mode == 'only_success' and trajectory['done'] == True):
                used_trajectories += 1
                x.extend(list(trajectory['trajectory']).copy()[:-shift])
                y.extend(list(trajectory['trajectory']).copy()[shift:])
        x = np.array(x)
        y = np.array(y)

        print(f'Using {used_trajectories} of {len(trajectories)} trajectories] '
              f'({round(100 * used_trajectories/len(trajectories),2)} %). Training on {len(x)} data points.')

        for network in self.ensemble:
            network.model.fit(x=x, y=y,epochs=epochs)


    def predict_goal(self, obs):


        predictions = []
        for network in self.ensemble:
            predictions.append(network.predict_goal(obs))

        predictions = np.array(predictions)
        goal = predictions.mean(axis=0)
        stdev = predictions.std(axis=0)
        print(f'       input = {obs}')
        print(f'        goal = {goal}')
        print(f'stdev (*100) = {100 * stdev}')

        # self.model.fit(np.array(x), np.array(y), epochs=2)
