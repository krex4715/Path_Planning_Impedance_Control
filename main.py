import gym
from custom_env import MobileRobotEnv
import numpy as np
import random
import buffer
import train
import matplotlib.pyplot as plt

import gc



num_obs=10

MAX_EPISODES = 3000
MAX_STEPS = 1000
MAX_BUFFER = 1000000

# env = MobileRobotEnv(num_obs,maxtime=MAX_STEPS,graphics=True)
env = MobileRobotEnv(num_obs,maxtime=MAX_STEPS)
print(env.observation_space.shape)
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]
print('----------------')
print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)
ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
threshold = 2000
trainer.load_models(load_dir='model_history_xy/',episode=threshold)


avg_reward = 0
success_count=0

loss_actors = []
loss_critics = []

training=False


plt.ion()  # Turn on interactive mode for matplotlib



for _ep in range(MAX_EPISODES):
    _ep = _ep+threshold
    obs = env.reset()
    state = np.float32(obs[0])
    print('EPISODE :- ', _ep)
    sum_reward=0
    count=0
    while True:
        count+=1
        env.render()
        if _ep%5 == 0:
            action = trainer.get_exploitation_action(state)
        else:
            action = trainer.get_exploration_action(state)

        
        new_obs, reward, done, info = env.step(action)
        # print(reward)
        new_state = np.float32(new_obs)

        

        sum_reward += reward
        
        
        if done:
            print('REWARD :- ', sum_reward)
            if training == True:
                break
        else:
            ram.add(state, action, reward, new_state)

        state = new_state
        iteration, loss_actor, loss_critic = trainer.optimize()
        if iteration%100==0:
            loss_actors.append(loss_actor)
            loss_critics.append(loss_critic)

        
    gc.collect()


    if _ep%100 == 0:
        trainer.save_models(save_dir='model_history_xy/' , episode_count=_ep)
        print(f'success " {success_count} / 100')
        success_count=0

plt.ioff() # Turn off interactive mode for matplotlib
env.close()
