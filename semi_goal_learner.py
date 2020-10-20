import pickle
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from doorman_env import Doorman

class PlannerSimulator:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.agent_pos = None

    def load_observation(self, obs):
        curr_keys = obs[2:]
        self.last_collected_key = sum(curr_keys)-1
        self.agent_pos = (obs[0], obs[1])


    def distance(self, x, y, conf):
        xt = np.multiply(x, conf)
        yt = np.multiply(y, conf)
        return 100 * np.square(xt-yt).mean()

    def state_quality(self, x, y, curr_state, desired_state, conf):
        curr_keys = curr_state[2:]
        last_collected_key = sum(curr_keys)-1
        if (x,y) in self.env.keys_pos:
            key_num = self.env.keys_pos[(x,y)]
            if key_num == last_collected_key + 1:
                last_collected_key = last_collected_key + 1
        obs_at_xy = [x,y] + [1]*(last_collected_key+1) + [0]*(self.env.n_keys-last_collected_key-1)
        obs_at_xy = np.array(obs_at_xy)
        quality = self.distance(obs_at_xy, desired_state, conf)
        return obs_at_xy, quality

    def create_map(self, curr_obs):
        desired_state, conf = self.model.predict_round(curr_obs, False)
        quality_map = np.zeros((self.env.size, self.env.size))

        for x in range(self.env.size):
            for y in range(self.env.size):
                pred_obs, dist = self.state_quality(x, y, curr_obs, desired_state, conf)
                quality_map[x][y] = 1 -dist
        return quality_map

    def find_best(self, curr_obs):
        desired_state, conf = self.model.predict_round(curr_obs, False)
        quality_map = np.zeros((self.env.size, self.env.size))
        curr_max = -1
        top = {'max': {'val': -1, 'pos': None}, 'second': {'val': -1, 'pos': None}}

        for x in range(self.env.size):
            for y in range(self.env.size):
                pred_obs, dist = self.state_quality(x, y, curr_obs, desired_state, conf)
                quality = 1 - dist
                quality_map[x][y] = quality
                if quality > top['max']['val']:
                    top['max']['val'] = quality
                    top['max']['pos'] = (x,y)
                elif quality > top['second']['val']:
                    top['second']['val'] = quality
                    top['second']['pos'] = (x,y)

        return top


    def go_to_xy(self, pos):
        def sgn(z):
            if z > 0:
                return 1
            elif z < 0:
                return -1
            else:
                return 0
        x, y = pos


        x_dir = sgn(x - self.agent_pos[0])
        y_dir = sgn(y - self.agent_pos[1])

        print(f'x = {x} y = {y} xdir = {x_dir} y_dir = {y_dir}')

        x_move = {-1: 'left', 1:'right'}
        y_move = {1: 'up', -1:'down'}

        if x_dir != 0 and y_dir != 0:
            use_x = random.randint(0, 1)
            if use_x:
                return x_move[x_dir]
            else:
                return y_move[y_dir]
        elif x_dir != 0:
            return x_move[x_dir]
        elif y_dir != 0:
            return y_move[y_dir]
        else:
            print('goal achieved')


    def show_map(self, curr_obs):
        map = self.create_map(curr_obs)
        sns.heatmap(map)
        plt.show()

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
        successful_trajectories = {}
        success_num = 0
        for _ in tqdm(range(n_trajectories)):
            trajectory, done, length = self.collect_random_trajectory(step_limit)
            if done:
                successful_trajectories[success_num] = {'trajectory': trajectory, 'done': done, 'length': length}
                success_num += 1
        print(f'Success rate = {round(100*success_num/n_trajectories, 2)} %.')
        if to_file:
            successful_trajectories_file = open(f'data/successful_tra.pkl', "wb")
            pickle.dump(successful_trajectories, successful_trajectories_file)
        return successful_trajectories

    def load_succesfull_trajectories(self):
        with open(f'data/successful_tra.pkl', 'rb') as f:
            good_tra = pickle.load(f)
        return good_tra


    def fit(self, trajectories, shift, epochs):
        x = []
        y = []

        for trajectory in trajectories.values():
            new_data_x = list(trajectory['trajectory']).copy()
            new_data_y = list(trajectory['trajectory']).copy()
            last_state_repeated = [new_data_y[-1]]*shift
            new_data_y.extend(last_state_repeated)
            new_data_y = new_data_y[shift:]
            assert len(new_data_x) == len(new_data_y)
            x.extend(new_data_x)
            y.extend(new_data_y)

        x = np.array(x)
        y = np.array(y)

        print(x.shape)
        print(y.shape)
        assert x.shape == y.shape

        print()

        self.model.fit(x=x, y=y, epochs=epochs)

    def reset(self):
        self.env.reset()

    def render(self):
        self.env.render()

    def episode_with_planner(self, n_episodes, step_limit):

        done = False
        reward = 0
        start_obs = self.env.reset()
        current_goal, confidence = self.model.predict_round(start_obs)
        print(f'current_goal = {current_goal} | conf = {confidence}')
        steps = 0


        # while not done and steps < step_limit:

