from dense_with_confidence import ConfidenceMLP
from semi_goal_learner import SemiGoalLearner
import pickle

SIZE = 10
KEYS = 3
EPISODE_LIMIT = 100

conf_mlp = ConfidenceMLP(5,[50,50])
goal_learner = SemiGoalLearner(SIZE,KEYS, conf_mlp)

#goal_learner.collect_random_trajectories(50000,EPISODE_LIMIT,True)
good_trajectories = goal_learner.load_succesfull_trajectories()


goal_learner.fit(good_trajectories, 20,5)


conf_mlp.predict([0,0,0,0,0], True)
