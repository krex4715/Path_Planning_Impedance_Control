import gym
import numpy as np
import random

class MobileRobotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.robot_size = 1
        self.max_speed = 5
        self.max_acc = 2
        self.time_step = 0.1
        self.target_pos = np.array([10, 10])
        self.max_distance = 15
        
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=self.max_distance, shape=(3, 2), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.robot_pos = np.random.uniform(low=self.robot_size, high=self.max_distance - self.robot_size, size=2)
        self.obstacle_pos = np.random.uniform(low=self.robot_size, high=self.max_distance - self.robot_size, size=(2, 2))
        self.obstacle_vel = np.random.uniform(low=-1, high=1, size=(2, 2))
        self.robot_vel = np.zeros(2)
        self.t = 0

        return self._get_observation()

    def _get_observation(self):
        target_dist = self.target_pos - self.robot_pos
        obs_pos = self.obstacle_pos - self.robot_pos
        return np.vstack((target_dist, obs_pos))

    def step(self, action):
        damping_coef, spring_coef = action

        target_force = spring_coef * (self.target_pos - self.robot_pos) - damping_coef * self.robot_vel

        obstacle_force = np.zeros(2)
        for pos, vel in zip(self.obstacle_pos, self.obstacle_vel):
            obs_dist = pos - self.robot_pos
            obs_force = spring_coef * obs_dist - damping_coef * (self.robot_vel - vel)
            obstacle_force += obs_force

        total_force = target_force - obstacle_force
        acc = np.clip(total_force, -self.max_acc, self.max_acc)
        self.robot_vel += acc * self.time_step
        self.robot_vel = np.clip(self.robot_vel, -self.max_speed, self.max_speed)
        self.robot_pos += self.robot_vel * self.time_step

        self.obstacle_pos += self.obstacle_vel * self.time_step

        done = False
        reward = 0

        if np.linalg.norm(self.target_pos - self.robot_pos) < self.robot_size:
            done = True
            reward = 100
        elif any(np.linalg.norm(self.robot_pos - pos) < self.robot_size for pos in self.obstacle_pos):
            done = True
            reward = -100
        elif self.t > 500:
            done = True
            reward = -10
        else:
            reward = -np.linalg.norm(self.target_pos - self.robot_pos)

        self.t += 1

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def q_learning(env, num_episodes, learning_rate, discount_factor, exploration_rate):
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    q_table = np.zeros((obs_dim, action_dim))

    for i_episode in range(num_episodes):
        observation = env.reset()
        obs_idx = tuple(observation.flatten())

        for t in range(500):
            if random.uniform(0, 1) < exploration_rate:
                action = env.action_space.sample()
            else:
                print(q_table)
                print(q_table[obs_idx])
                action = q_table[obs_idx].argmax()

            next_observation, reward, done, _ = env.step(action)
            next_obs_idx = tuple(next_observation.flatten())
            target = reward + discount_factor * np.max(q_table[next_obs_idx])
            q_table[obs_idx][action] += learning_rate * (target - q_table[obs_idx][action])

            observation = next_observation
            obs_idx = next_obs_idx

            if done:
                break

        if (i_episode + 1) % 100 == 0:
            print(f"Episode {i_episode + 1}/{num_episodes} finished.")

    return q_table

def test_agent(env, q_table):
    observation = env.reset()
    obs_idx = tuple(observation.flatten())
    total_reward = 0

    for t in range(500):
        action = q_table[obs_idx].argmax()
        observation, reward, done, _ = env.step(action)
        obs_idx = tuple(observation.flatten())
        total_reward += reward

        if done:
            break

    return total_reward

num_tests = 10
test_rewards = []




if __name__=="__main__":

    env = MobileRobotEnv()
    num_episodes = 1000
    learning_rate = 0.1
    discount_factor = 0.99
    exploration_rate = 0.1

    q_table = q_learning(env, num_episodes, learning_rate, discount_factor, exploration_rate)



    for _ in range(num_tests):
        test_rewards.append(test_agent(env, q_table))

    print(f"Average test reward: {np.mean(test_rewards)}")
