import pickle
import random
import numpy as np
from tqdm import tqdm

from doorman_env import Doorman

class SemiGoalLearner:
    def __init__(self, size, n_keys, model):
        self.size = size
        self.n_keys = n_keys
        self.env = Doorman(size, n_keys)
        self.model = model


    def collect_random_trajectory(self, step_limit):
        done = False
        obs = self.env.reset()
        step = 0
        trajectory = [obs]
        while not done and step < step_limit:
            obs, rew, done, info = self.env.step(random.randint(0,3))
            trajectory.append(obs)
            step += 1
        return tuple(trajectory), done, len(trajectory)

    def collect_random_trajectories(self, n_trajectories, step_limit, to_file=False):
        trajectories = {}
        successful_trajectories = {}
        success_num = 0
        for num in tqdm(range(n_trajectories)):
            trajectory, done, length = self.collect_random_trajectory(step_limit)
            if done:
                successful_trajectories[success_num] = {'trajectory': trajectory, 'done': done, 'length': length}
                success_num += 1
            trajectories[num] = {'trajectory': trajectory, 'done': done, 'length': length}
        print(f'Success rate = {round(100*success_num/n_trajectories, 2)} %.')
        if to_file:
            successful_trajectories_file = open(f'data/successful_tra.pkl', "wb")
            pickle.dump(successful_trajectories, successful_trajectories_file)
        return trajectories, successful_trajectories

    def load_succesfull_trajectories(self):
        with open(f'data/successful_tra.pkl', 'rb') as f:
            good_tra = pickle.load(f)
        return good_tra

    def fit(self, trajectories, shift, epochs):
        x = []
        y = []

        used_trajectories = 0
        for trajectory in trajectories.values():
            x.extend(list(trajectory['trajectory']).copy()[:-shift])
            y.extend(list(trajectory['trajectory']).copy()[shift:])
        x = np.array(x)
        y = np.array(y)

        self.model.fit(x=x, y=y, epochs=epochs)

    def reset(self):
        self.env.reset()

    def render(self):
        self.env.render()