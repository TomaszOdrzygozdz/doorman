from dense_with_confidence import ConfidenceMLP
from doorman_env import Doorman
from semi_goal_learner import SemiGoalLearner, PlannerSimulator

SIZE = 10
KEYS = 3
EPISODE_LIMIT = 100
N_TRAJECTORIES = 300000
SAVE_TRAJECTORIES = False

env = Doorman(SIZE,KEYS)
conf_mlp = ConfidenceMLP(5,[100,100])
goal_learner = SemiGoalLearner(SIZE,KEYS, conf_mlp)
planner = PlannerSimulator(env, conf_mlp)

goal_x = 5
goal_y = 7
start = env.reset()
planner.load_observation(start)

while True:
    print(f'agents pos = {planner.agent_pos}')
    act = planner.go_to_xy((5,6))
    obs, _, _, _ = env.step(act)
    print(f'env obs = {obs}')
    planner.load_observation(obs)