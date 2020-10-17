from typing import List

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam


class GoalNetwork:

    def __init__(self, n_keys, hidden_layers):

        input = Input(batch_shape=(None, 2 + n_keys))
        layer = input

        for hidden_size, num in enumerate(hidden_layers):
            layer = Dense(hidden_size, activation='relu')(layer)

        output = Dense(2 + n_keys)(layer)

        self.model = Model(inputs=input, outputs=output)
        self.model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy', 'mean_squared_error'])

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


    def fit_trajectories(self, trajectories, epochs, mode='only_success'):

        x = []
        y = []

        succesfull_trajectories = 0
        for trajectory in trajectories.values():

            if mode == 'all' or (mode == 'only_success' and trajectory['done'] == True):
                if mode == 'only_success':
                    succesfull_trajectories += 1
                x.extend(list(trajectory['trajectory']).copy()[:-1])
                y.extend(list(trajectory['trajectory']).copy()[1:])

        print(f'Using {succesfull_trajectories} of {len(trajectories)} trajectories '
              f'({round(100 * succesfull_trajectories/len(trajectories),2)} %). Training on {len(x)} data points.')

        for network in self.ensemble:
            network.model.fit(x=x, y=y,epochs=epochs)



        # self.model.fit(np.array(x), np.array(y), epochs=2)
