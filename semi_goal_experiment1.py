from dense_with_confidence import ConfidenceMLP
from semi_goal_learner import SemiGoalLearner
import pickle

SIZE = 10
KEYS = 3
EPISODE_LIMIT = 100
N_TRAJECTORIES = 250000
SAVE_TRAJECTORIES = False

conf_mlp = ConfidenceMLP(5,[50,50])
goal_learner = SemiGoalLearner(SIZE,KEYS, conf_mlp)

if SAVE_TRAJECTORIES:
    good_trajectories = goal_learner.collect_random_trajectories(N_TRAJECTORIES,EPISODE_LIMIT,True)
else:
    good_trajectories = goal_learner.load_succesfull_trajectories()

goal_learner.fit(good_trajectories, 10,5)


conf_mlp.predict([0,0,0,0,0], True)
conf_mlp.predict([0,0,1,0,0], True)
conf_mlp.predict([1,2,0,0,0], True)
conf_mlp.predict([0,0,1,1,0], True)
conf_mlp.predict([0,3,1,1,0], True)
