# env_v4.py
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection

class MobileRobotEnv(gym.Env):
    def __init__(self, num_obs=6,maxtime=1000):
        super().__init__()

        self.robot_size = 0.5
        self.robot_mass = 0.3
        self.max_speed = 6
        self.max_acc = 2
        self.time_step = 0.05
        
        self.rot_seed = 2*np.pi*np.random.uniform()
        self.target_pos = np.float32([10 + 7*np.cos(self.rot_seed), 10 + 7*np.sin(self.rot_seed)])
        self.max_distance = 20
        self.num_obs = num_obs
        self.detection_range = 5
        self.desired_positions = np.zeros((self.num_obs, 2))
        self.obstacle_damping_coef = np.zeros(num_obs)
        self.obstacle_spring_coef = np.zeros(num_obs)
        self.obs_force = 0
        self.target_damping_coef = 0
        self.target_spring_coef = 0
        self.maxtime=maxtime


        self.action_space = gym.spaces.Box(
            low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-self.max_distance, high=self.max_distance, shape=(2 * (self.num_obs + 1),), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.rot_seed = 2*np.pi*np.random.uniform()
        self.robot_pos = np.random.uniform(low=self.robot_size, high=self.max_distance - self.robot_size, size=2)
        self.obstacle_pos = np.random.uniform(low=self.robot_size,
                                              high=self.max_distance - self.robot_size,
                                              size=(self.num_obs, 2),
                                              )
        self.obstacle_vel = np.random.normal(loc=0.0, scale=1.0, size=(10, 2))
        self.robot_vel = np.zeros(2)
        self.t = 0
        self.target_area = 4*self.robot_size
        return self._get_observation()

    def _get_observation(self):
        self.target_dist = self.target_pos - self.robot_pos
        obs_pos = self.obstacle_pos - self.robot_pos
        dists = np.linalg.norm(obs_pos, axis=1)
        self.count_in_area = np.count_nonzero(dists < self.detection_range)

        sorted_indices = np.argsort(dists)
        self.closest_indices = sorted_indices[:self.count_in_area]

        observation = np.zeros((2 * (self.num_obs + 1),))
        observation[-2:] = self.target_dist
        for i in range(self.count_in_area):
            observation[2 * i: 2 * (i + 1)] = obs_pos[self.closest_indices[i]]
        for i in range(self.count_in_area, self.num_obs):
            observation[2 * i: 2 * (i + 1)] = [0, 0]
            return observation, self.count_in_area

    def step(self, action):
        self.robot_vel = np.clip(action, -self.max_speed, self.max_speed)

        self.obstacle_vel = 2*np.random.uniform(low=-2, high=2, size=(self.num_obs, 2))
        self.target_vel = 2*np.random.uniform(low=-2, high=2, size=(2))

        self.robot_pos += self.robot_vel * self.time_step
        self.robot_pos = np.clip(self.robot_pos, 0, self.max_distance - self.robot_size)

        self.obstacle_pos += self.obstacle_vel * self.time_step
        self.obstacle_pos = np.clip(self.obstacle_pos, 0, self.max_distance - self.robot_size)

        self.target_pos = np.float32([10 + 7*np.cos((2*np.pi*self.t/300 + self.rot_seed)), 10 + 7*np.sin((2*np.pi*self.t/300 + self.rot_seed))])

        done = False
        reward = 0
        success=False

        if np.linalg.norm(self.target_pos - self.robot_pos) < self.target_area:
            success=True
            if np.linalg.norm(self.target_pos - self.robot_pos) < self.robot_size:
                done = True
            reward = 40
        
        elif np.linalg.norm(self.target_pos - self.robot_pos) < self.detection_range:
            reward = 10-np.linalg.norm(self.target_pos - self.robot_pos)
        
        elif any(np.linalg.norm(self.robot_pos - pos) < self.robot_size for pos in self.obstacle_pos):
            done = True
            reward = -10000
            success=False
        
        elif self.t > self.maxtime:
            done = True
            reward = 10
            success=False
        
        else:
            reward = -np.linalg.norm(self.target_pos - self.robot_pos)

        self.t += 1
        return self._get_observation(), reward, done, {}, self.count_in_area , success


    def render(self):
        if not hasattr(self, 'fig') or self.fig.canvas.manager.window is None:
            self.fig, self.ax = plt.subplots(figsize=(15, 15))
            plt.ion()
        self.ax.clear()
        self.ax.set_xlim(0-3, self.max_distance+3)
        self.ax.set_ylim(0-3, self.max_distance+3)
        self.ax.set_aspect('equal')

        # Draw the robot
        robot = Circle(self.robot_pos, self.robot_size, color='b', label="Robot")
        self.ax.add_patch(robot)

        # Draw the target
        self.ax.scatter(self.target_pos[0], self.target_pos[1], c='red', marker='x', s=100, label='Target')


        # Draw the obstacles
        for pos, vel, desired_pos in zip(self.obstacle_pos, self.obstacle_vel, self.desired_positions):
            self.ax.scatter(pos[0], pos[1], c='black', marker='s', s=100, label='Obstacle')

        # # Draw the fixed, large rectangular obstacle
        # rect_obstacle = Rectangle((5, 5), 2, 5, color='m', label="Fixed Obstacle")
        # self.ax.add_patch(rect_obstacle)

        # Draw the legend
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        # Draw detection range circle
        detection_circle = Circle(self.robot_pos, self.detection_range, fill=False, linestyle='dashed', color='gray')
        target_area_circle = Circle(self.target_pos, self.target_area, fill=False, linestyle='dashed', color='gray')

        self.ax.add_patch(detection_circle)
        self.ax.add_patch(target_area_circle)
        self.ax.quiver(
                        self.robot_pos[0],
                        self.robot_pos[1],
                        self.robot_vel[0],
                        self.robot_vel[1],
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        color="blue",
                        )

        # Draw lines between robot, obstacles, and target with colors representing force strength
        lines = []
        colors = []
        for idx, (pos, vel) in enumerate(zip(self.obstacle_pos, self.obstacle_vel)):
            obs_dist = pos - self.robot_pos
            if np.linalg.norm(obs_dist) < self.detection_range:
                # Get the order of the current obstacle in closest_indices
                if idx in self.closest_indices:
                    obstacle_order = np.where(self.closest_indices == idx)[0][0] + 1

                self.ax.text(pos[0] + 0.5, pos[1], f"{obstacle_order}\nS: {self.obstacle_spring_coef[idx]:.2f}\nD: {self.obstacle_damping_coef[idx]:.2f}", fontsize=12, color='black')
                            
                lines.append([self.robot_pos, pos])
                colors.append(self.obs_force)
            else:
                self.ax.text(pos[0] + 0.5, pos[1], f"Out", fontsize=12, color='black')


        lines.append([self.robot_pos, self.target_pos])
        
        line_collection = LineCollection(lines, cmap='twilight_shifted_r', linewidths=1)
        self.ax.add_collection(line_collection)
        self.ax.text(self.target_pos[0] + 0.5, self.target_pos[1], f"D: {self.target_damping_coef:.2f}\nS: {self.target_spring_coef:.2f}", fontsize=12, color='red')




        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)


