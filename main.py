import gym
from env_v2 import MobileRobotEnv
import numpy as np
import random
import matplotlib.pyplot as plt




num_obs=50
env = MobileRobotEnv(num_obs)

num_tests = 10
test_rewards = []

plt.ion()  # Turn on interactive mode for matplotlib



for _ in range(num_tests):
    obs = env.reset()
    env.render()
    total_reward = 0

    while True:
        action = env.action_space.sample()
        # action=[0.3,0.8]*num_obs+[0.5,2]
        obs, reward, done, _,count_in_area = env.step(action)
        # print(count_in_area)
        total_reward += reward
        env.render()
        if done:
            break

    test_rewards.append(total_reward)
    plt.pause(1)  # Add a pause between tests to better visualize each test
plt.ioff() # Turn off interactive mode for matplotlib
env.close()
print(f"Average test reward: {np.mean(test_rewards)}")