from keras_network import GoalNetworkEnsemble
from semi_goal_learner import SemiGoalLearner

size = 10

uu = SemiGoalLearner(size,3)

st = uu.collect_random_trajectories(2000,500)
d1 = GoalNetworkEnsemble(size,3,[100,100],5)
d1.fit_trajectories(st,50,2)

def xy(x, y, keys):
    x_one_hot = [0]*size
    x_one_hot[x] = 1
    y_one_hot = [0]*size
    y_one_hot[y] = 1
    return x_one_hot + y_one_hot + keys


d1.predict_goal(xy(0,0,[0,0,0]))
d1.predict_goal(xy(0,1,[0,0,0]))
d1.predict_goal(xy(0,0,[1,0,0]))
d1.predict_goal(xy(0,4,[1,0,0]))
d1.predict_goal(xy(3,1,[1,1,0]))




