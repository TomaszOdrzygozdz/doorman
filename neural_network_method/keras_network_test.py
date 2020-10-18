from neural_network_method.keras_network import GoalNetwork
from semi_goal_learner import SemiGoalLearner

size = 6
n_keys = 3

semi_goal_learner = SemiGoalLearner(size,n_keys)

random_paths = semi_goal_learner.collect_random_trajectories(100,200)
net = GoalNetwork(size, n_keys,[50,50])

net.fit_trajectories(random_paths,5,2)



# d1 = GoalNetworkEnsemble(size,3,[100,100],5)
# d1.fit_trajectories(st,50,2)
#
# def xy(x, y, keys):
#     x_one_hot = [0]*size
#     x_one_hot[x] = 1
#     y_one_hot = [0]*size
#     y_one_hot[y] = 1
#     return x_one_hot + y_one_hot + keys
#
#
# d1.predict_goal(xy(0,0,[0,0,0]))
# d1.predict_goal(xy(0,1,[0,0,0]))
# d1.predict_goal(xy(0,0,[1,0,0]))
# d1.predict_goal(xy(0,4,[1,0,0]))
# d1.predict_goal(xy(3,1,[1,1,0]))




