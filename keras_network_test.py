import numpy as np

from keras_network import GoalNetwork, GoalNetworkEnsemble
from semi_goal_learner import SemiGoalLearner

uu = SemiGoalLearner(5,3)

st = uu.collect_random_shortened_trajectories(5,500,50)
d = GoalNetworkEnsemble(3,[10,10],5)
d.fit_trajectories(st,2)


# tt, dd = uu.collect_random_trajectory(50000)
# tt2, dd2 = uu.collect_random_trajectory(40000)
# print(f'dd = {dd}')
#
# d = GoalNetwork(3,[50,50,50])
# e = GoalNetwork(3,[50,50,50])
#
# d.fit_trajectory(tt)
# e.fit_trajectory(tt2)
#
# y1 = d.predict_goal(np.array([[2,2,1,0,0]]))
# print(y1)
# y2 = e.predict_goal(np.array([[2,2,1,0,0]]))
# print(y2)
# print("==")
# y1 = d.predict_goal(np.array([[2,2,1,0,0]]))
# print(y1)
# y2 = e.predict_goal(np.array([[2,2,1,0,0]]))
# print(y2)