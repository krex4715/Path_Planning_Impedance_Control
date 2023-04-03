from custom_env import MobileRobotEnv
import numpy as np
import random
import buffer
import train
import matplotlib.pyplot as plt


import gc



num_obs=20

MAX_EPISODES = 3000
MAX_STEPS = 1000
MAX_BUFFER = 1000000


# env = MobileRobotEnv(num_obs,maxtime=MAX_STEPS,graphics=True)

env = MobileRobotEnv(num_obs,maxtime=MAX_STEPS)


plt.ion()



for _ep in range(MAX_EPISODES):
    obs = env.reset()
    state = np.float32(obs[0])
    count_in_area = obs[1]
    print('EPISODE :- ', _ep)
    sum_reward=0
    count=0
    while True:
        count+=1
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print('STATE :- ', next_state)
        next_state = np.float32(next_state)
        sum_reward+=reward
        if done:
            print('REWARD :- ', sum_reward)
            break
        state = next_state
    


plt.ioff()
env.close()