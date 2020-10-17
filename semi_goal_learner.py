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

    def shorten_trajectories(self, trajectories, every_k):
        shortened_trajectories = {}
        for num, trajectory in trajectories.items():
            shortened_trajectories[num] = {
                'trajectory': self.shorten_trajectory(trajectory['trajectory'], every_k),
                'done': trajectories[num]['done']
            }
        return shortened_trajectories

    def shorten_trajectory(self, trajectory, every_k):
        short_trajectory = list(trajectory)[::every_k] + [np.array(list(trajectory)[-1])]
        return tuple(short_trajectory)

    def collect_random_shortened_trajectories(self, num_shortened_trajectories, step_limit, every_k):
        full_trajectories = self.collect_random_trajectories(num_shortened_trajectories, step_limit)
        return self.shorten_trajectories(full_trajectories, every_k)