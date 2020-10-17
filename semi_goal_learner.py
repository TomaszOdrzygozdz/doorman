import random
import numpy as np
from tqdm import tqdm

from doorman_env import Doorman


class SemiGoalLearner:
    def __init__(self, size, n_keys):
        self.size = size
        self.n_keys = n_keys
        self.env = Doorman(size, n_keys)

    def collect_random_trajectory(self, step_limit):
        done = False
        obs = self.env.reset()
        step = 0
        trajectory = [obs]
        while not done and step < step_limit:
            obs, rew, done, info = self.env.step(random.randint(0,3))
            trajectory.append(obs)
            step += 1
        return tuple(trajectory), done

    def collect_random_trajectories(self, n_trajectories, step_limit=200):
        trajectories = {}
        for num in tqdm(range(n_trajectories)):
            trajectory, done = self.collect_random_trajectory(step_limit)
            trajectories[num] = {'trajectory': trajectory, 'done': done}
        return trajectories