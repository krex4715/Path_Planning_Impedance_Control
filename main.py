import gym
from env_v4 import MobileRobotEnv
import numpy as np
import random
import matplotlib.pyplot as plt




num_obs=30
env = MobileRobotEnv(num_obs)

num_tests = 20
test_rewards = []

plt.ion()  # Turn on interactive mode for matplotlib



for _ in range(num_tests):
    obs = env.reset()
    env.render()
    total_reward = 0

    while True:
        action = env.action_space.sample()
        # # Constant Agent
        action=[1,1]*num_obs+[1,3]
        obs, reward, done, _,count_in_area = env.step(action)
        print(print(obs))
        total_reward += reward
        env.render()
        if done:
            break

    test_rewards.append(total_reward)
    plt.pause(1)  # Add a pause between tests to better visualize each test
plt.ioff() # Turn off interactive mode for matplotlib
env.close()
print(f"Average test reward: {np.mean(test_rewards)}")