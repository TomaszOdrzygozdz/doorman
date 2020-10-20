from doorman_env import Doorman
from semi_goal_learner import PlannerSimulator
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

env = Doorman(5,3)
env.reset()


planner = PlannerSimulator(env, None)

print(env.keys_pos)

# for x in range(5):
#     for y in range(5):
#         predicted_obs, quality = planner.state_quality(x, y, [0,0,0,0,0], [1,2,1,0,0], [0,0,1,1,1])
#         print(f'x = {x} y = {y} | obs_at_xy = {predicted_obs} dist = {quality}')

