from dense_with_confidence import ConfidenceMLP
from doorman_env import Doorman
from semi_goal_learner import SemiGoalLearner, PlannerSimulator
import pickle

SIZE = 10
KEYS = 3
EPISODE_LIMIT = 100
N_TRAJECTORIES = 300000
SAVE_TRAJECTORIES = False

env = Doorman(SIZE,KEYS)
env.reset()
conf_mlp = ConfidenceMLP(5,[100,100])
goal_learner = SemiGoalLearner(SIZE,KEYS, conf_mlp)
planner = PlannerSimulator(env, conf_mlp)

if SAVE_TRAJECTORIES:
    good_trajectories = goal_learner.collect_random_trajectories(N_TRAJECTORIES,EPISODE_LIMIT,True)
else:
    good_trajectories = goal_learner.load_succesfull_trajectories()


print(f'Keys_positions = {env.keys}')

#goal_learner.fit(good_trajectories, 25,1)
conf_mlp.predict_round([0,0,0,0,0], True)

goal_learner.episode_with_planner(1,2)



