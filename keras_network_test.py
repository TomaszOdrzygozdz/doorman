from keras_network import GoalNetworkEnsemble
from semi_goal_learner import SemiGoalLearner

uu = SemiGoalLearner(8,3)

st = uu.collect_random_shortened_trajectories(10000,500,10)
d1 = GoalNetworkEnsemble(3,[250,250],3)
d1.fit_trajectories(st,2)

d1.predict_goal([0,0,0,0,0])
d1.predict_goal([1,2,0,0,0])

d1.predict_goal([0,0,1,0,0])
d1.predict_goal([1,2,1,0,0])



