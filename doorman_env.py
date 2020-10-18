import random

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class Doorman:

    def __init__(self, size, n_keys):
        self.size = size
        assert n_keys <= 6, "You can have up to 6 keys."
        self.n_keys = n_keys
        self.last_collected_key = -1
        self.action_lookup = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

    def reset(self):
        self.agent_pos = self._random_pos()
        self.keys = {}
        self.keys_pos = {}
        self.keys_colors = {}
        self.last_collected = -1
        self.keys_collected = []
        self.done = False
        self._place_keys()
        self.key_colors =  ['red', 'green', 'blue', 'yellow', 'brown', 'gray']
        return self.create_observation()

    def create_observation(self):
        return self.create_observation_xyk()

    def create_observation_xy(self):
        return np.array(self.normalized_position() + tuple([key in self.keys_collected for key in range(self.n_keys)]))

    def create_observation_xyk(self):
        keys_collected = tuple([key in self.keys_collected for key in range(self.n_keys)])
        return [np.asarray(self.agent_pos[0]).astype('float32')] + [np.asarray(self.agent_pos[1]).astype('float32')]  \
               + [np.asarray(key).astype('float32') for key in keys_collected]

    def create_observation_one_hot(self):

        one_hot_x = [0] * self.size
        one_hot_x[self.agent_pos[0]] = 1
        one_hot_y = [0] * self.size
        one_hot_y[self.agent_pos[1]] = 1

        return np.array(tuple(one_hot_x) + tuple(one_hot_y)
                        + tuple([key in self.keys_collected for key in range(self.n_keys)]))

    def normalized_position(self):
        return tuple(np.array(self.agent_pos)/self.size)

    def step(self, action):

        assert self.done == False, 'Episode ended.'
        info = {'wall_hit' : [], 'keys_collected': []}
        reward = 0

        if type(action) is int:
            move = action
        elif type(action) is str:
            move = self.action_lookup[action]

        self.moves = {0: (0,1), 1: (0,-1), 2: (-1,0), 3: (1,0)}

        new_pos = tuple([self.agent_pos[i] + self.moves[move][i] for i in range(2)])
        if min(new_pos) >= 0 and max(new_pos) < self.size:
            self.agent_pos = new_pos
            info['wall_hit'].append(False)
        else:
            info['wall_hit'].append(True)

        if self.agent_pos in self.keys_pos:
            if self.keys_pos[self.agent_pos] == self.last_collected + 1:
                key_num = self.keys_pos[self.agent_pos]
                self.last_collected = self.last_collected + 1
                del self.keys_pos[self.agent_pos]
                del self.keys[key_num]
                self.keys_collected.append(key_num)

        info['keys_collected'] = self.keys_collected

        if len(info['keys_collected']) == self.n_keys:
            self.done = True
            reward = 1

        return self.create_observation(), reward, self.done, info

    def _random_pos(self):
        return (random.randint(0,self.size-1), random.randint(0,self.size-1))

    def _place_keys(self):
        def try_placing():
            positions = {self.agent_pos}
            for key_num in range(self.n_keys):
                key_pos = self._random_pos()
                if key_pos not in positions:
                    positions.add(key_pos)
                else:
                    return False
                self.keys.update({key_num: key_pos})
                self.keys_pos.update({key_pos: key_num})
            return True

        correctly_placed = False
        while not correctly_placed:
            correctly_placed = try_placing()

    def render(self):

        x = [self.agent_pos[0]]
        y = [self.agent_pos[1]]
        colors = ['black']
        for key_num, key_pos in self.keys.items():
            x.append(key_pos[0])
            y.append(key_pos[1])
            colors.append(self.key_colors[key_num])

        area = 100
        plt.grid(b=True, axis='both')
        plt.ylim(-0.1,self.size+0.1)
        plt.xlim(-0.1, self.size+0.1)
        plt.yticks(list(range(self.size+1)))
        plt.xticks(list(range(self.size+1)))
        plt.scatter(x, y, s=area, c=colors, alpha=1)
        plt.show()