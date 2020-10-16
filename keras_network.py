from typing import List

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

from semi_goal_learning import SemiGoalLearner


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

    def fit_trajectories(self, trajectory: List, epochs):
        x = np.array(trajectory.copy())[:-1]
        y = np.array(trajectory.copy())[1:]
        self.model.fit(np.array(x), np.array(y), epochs=2)



uu = SemiGoalLearner(45,3)
tt, dd = uu.collect_random_trajectory(50000)
tt2, dd2 = uu.collect_random_trajectory(40000)
print(f'dd = {dd}')

d = GoalNetwork(3,[50,50,50])
e = GoalNetwork(3,[50,50,50])

d.fit_trajectory(tt)
e.fit_trajectory(tt2)

y1 = d.predict_goal(np.array([[2,2,1,0,0]]))
print(y1)
y2 = e.predict_goal(np.array([[2,2,1,0,0]]))
print(y2)
print("==")
y1 = d.predict_goal(np.array([[2,2,1,0,0]]))
print(y1)
y2 = e.predict_goal(np.array([[2,2,1,0,0]]))
print(y2)