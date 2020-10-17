from keras_network import GoalNetworkEnsemble
from semi_goal_learner import SemiGoalLearner

uu = SemiGoalLearner(10,3)

st = uu.collect_random_trajectories(5000,500)
d1 = GoalNetworkEnsemble(3,[100,100],5)
d1.fit_trajectories(st,100,2)

d1.predict_goal([0,0,0,0,0])
d1.predict_goal([0.1,0.2,0,0,0])

d1.predict_goal([0,0,1,0,0])
d1.predict_goal([0.1,0.2,1,0,0])



