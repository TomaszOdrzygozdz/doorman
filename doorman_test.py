from doorman_env import Doorman

d = Doorman(2,3)

obs = d.reset()
print(obs)

print(d.agent_pos)
print(d.keys)
print(f'key_pos = {d.keys_pos}')
d.render()

while True:
    print("=======================================")
    act = input()
    print(f'self.agent_pos = {d.agent_pos}')
    obs, reward, done, info = d.step(act)
    print(f'{d.agent_pos} obs={obs} info = {info} reward = {reward} done = {done}')
    d.render()