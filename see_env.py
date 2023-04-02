import gym
from env_imp import MobileRobotEnv
import numpy as np
import random
import matplotlib.pyplot as plt




num_obs=10
env = MobileRobotEnv(num_obs,maxtime=2000,graphics=True)

MAX_EPISODES=3000

num_tests = 20
test_rewards = []

plt.ion()  # Turn on interactive mode for matplotlib

success_count=0

for _ep in range(MAX_EPISODES):
    obs = env.reset()
    env.render()
    total_reward = 0

    while True:
        action = env.action_space.sample()
        # # Constant Agent
        # action=[1,5]*num_obs+[2,5]
        obs, reward, done, _,count_in_area,success = env.step(action)
        # print(obs)
        # print(reward)
        total_reward += reward
        env.render()
        # print(done)
        if done:
            break

    test_rewards.append(total_reward)
    if _ep%10 == 0:
        print(f'success " {success_count} / 100')
        success_count=0
    # plt.pause(1)  # Add a pause between tests to better visualize each test
plt.ioff() # Turn off interactive mode for matplotlib
env.close()
print(f"Average test reward: {np.mean(test_rewards)}")