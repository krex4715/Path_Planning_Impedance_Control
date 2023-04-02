import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


class MobileRobotEnv(gym.Env):
    def __init__(self, num_obs=2):
        super().__init__()

        self.robot_size = 0.001
        self.max_speed = 5
        self.max_acc = 2
        self.time_step = 0.1
        self.target_pos = np.array([10, 10])
        self.max_distance = 15
        self.detection_range = 3
        self.num_obs = num_obs

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=self.max_distance, shape=(3, 2), dtype=np.float32)

        self.obstacle_damping_coef=0
        self.obstacle_spring_coef=0
        
        self.target_damping_coef=0
        self.target_spring_coef=0


        self.reset()



    def reset(self):
        self.robot_pos = np.random.uniform(low=self.robot_size, high=self.max_distance - self.robot_size, size=2)
        self.obstacle_pos = np.random.uniform(low=self.robot_size, high=self.max_distance - self.robot_size, size=(self.num_obs, 2))
        self.obstacle_vel = np.random.uniform(low=-1, high=1, size=(self.num_obs, 2))
        self.robot_vel = np.zeros(2)
        self.t = 0
        return self._get_observation()

    def _get_observation(self):
        target_dist = self.target_pos - self.robot_pos
        obs_pos = self.obstacle_pos - self.robot_pos
        return np.vstack((target_dist, obs_pos))

    def step(self, action):
        self.target_damping_coef, self.target_spring_coef, self.obstacle_damping_coef, self.obstacle_spring_coef = action

        target_force = self.target_spring_coef * (self.target_pos - self.robot_pos) - self.target_damping_coef * self.robot_vel

        obstacle_force = np.zeros(2)
        for pos, vel in zip(self.obstacle_pos, self.obstacle_vel):
            obs_dist = pos - self.robot_pos
            if np.linalg.norm(obs_dist) < self.detection_range:
                obs_force = self.obstacle_spring_coef * obs_dist - self.obstacle_damping_coef * (self.robot_vel - vel)
                obstacle_force += obs_force

        total_force = target_force - obstacle_force
        acc = np.clip(total_force, -self.max_acc, self.max_acc)
        self.robot_vel += acc * self.time_step
        self.robot_vel = np.clip(self.robot_vel, -self.max_speed, self.max_speed)
        self.robot_pos += self.robot_vel * self.time_step

        self.obstacle_pos += self.obstacle_vel * self.time_step

        done = False
        reward =0

        if np.linalg.norm(self.target_pos - self.robot_pos) < self.robot_size:
            done = True
            reward = 100
        elif any(np.linalg.norm(self.robot_pos - pos) < self.robot_size for pos in self.obstacle_pos):
            done = True
            reward = -100
        elif self.t > 300:
            done = True
            reward = -10
        else:
            reward = -np.linalg.norm(self.target_pos - self.robot_pos)

        self.t += 1

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'fig') or self.fig.canvas.manager.window is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.max_distance)
        self.ax.set_ylim(0, self.max_distance)
        
        # Draw target
        self.ax.scatter(self.target_pos[0], self.target_pos[1], c='red', marker='x', s=100, label='Target')

        # Draw robot
        self.ax.scatter(self.robot_pos[0], self.robot_pos[1], c='blue', marker='o', s=100, label='Robot')

        # Draw obstacles
        for idx, pos in enumerate(self.obstacle_pos):
            self.ax.scatter(pos[0], pos[1], c='black', marker='s', s=100, label='Obstacle' if idx == 0 else None)

        # Draw detection range circle
        detection_circle = Circle(self.robot_pos, self.detection_range, fill=False, linestyle='dashed', color='gray')
        self.ax.add_patch(detection_circle)

        # Draw lines between robot, obstacles, and target with colors representing force strength
        lines = []
        colors = []
        for pos, vel in zip(self.obstacle_pos, self.obstacle_vel):
            obs_dist = pos - self.robot_pos
            if np.linalg.norm(obs_dist) < self.detection_range:
                obs_force = np.linalg.norm(self.obstacle_spring_coef * obs_dist - self.obstacle_damping_coef * (self.robot_vel - vel))
                lines.append([self.robot_pos, pos])
                colors.append(obs_force)

        lines.append([self.robot_pos, self.target_pos])
        target_force = np.linalg.norm(self.target_spring_coef * (self.target_pos - self.robot_pos) - self.target_damping_coef * self.robot_vel)
        colors.append(target_force)

        line_collection = LineCollection(lines, cmap='coolwarm', linewidths=2)
        line_collection.set_array(np.array(colors))
        self.ax.add_collection(line_collection)

        # Draw legend
        self.ax.legend()

        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass





