import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


class MobileRobotEnv(gym.Env):
    def __init__(self, num_obs=6):
        super().__init__()

        self.robot_size = 0.3
        self.max_speed = 5
        self.max_acc = 2
        self.time_step = 0.05
        self.target_pos = np.random.uniform(low=0, high=15, size=(2))
        self.max_distance = 15
        self.num_obs = num_obs
        self.detection_range = 3

        self.action_space = gym.spaces.Box(
            low=0, high=2, shape=(2 * (self.num_obs + 2),), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_distance, shape=(2 * (self.num_obs + 1),), dtype=np.float32
        )
        
        self.obstacle_damping_coef=np.zeros(num_obs)
        self.obstacle_spring_coef=np.zeros(num_obs)
        
        self.target_damping_coef=0
        self.target_spring_coef=0
        self.obs_force=0
        self.count_in_area=0

        self.reset()

    def reset(self):
        self.robot_pos = np.random.uniform(low=self.robot_size, high=self.max_distance - self.robot_size, size=2)
        self.obstacle_pos = np.random.uniform(low=self.robot_size,
                                                high=self.max_distance - self.robot_size,
                                                size=(self.num_obs, 2),
                                                )
        self.obstacle_vel = np.random.normal(loc=0.0, scale=1.0, size=(10, 2))
        self.robot_vel = np.zeros(2)
        self.t = 0
        return self._get_observation()


    def _get_observation(self):
        self.target_dist = self.target_pos - self.robot_pos
        obs_pos = self.obstacle_pos - self.robot_pos
        dists = np.linalg.norm(obs_pos, axis=1)
        self.count_in_area = np.count_nonzero(dists<self.detection_range)
        
        sorted_indices = np.argsort(dists)
        closest_indices = sorted_indices[: self.num_obs]

        observation = np.zeros((2 * (self.num_obs + 1),))
        observation[-2:] = self.target_dist
        for i, idx in enumerate(closest_indices):
            if i<=self.count_in_area:
                observation[2 * i : 2 * (i + 1)] = obs_pos[idx]
            else:
                observation[2 * i : 2 * (i + 1)] = [0,0]
        return observation ,self.count_in_area

    def step(self, action):
        # 0~10 detection_range
        self.detection_range = action[-1]*5
        
        action = action[:-1]
        self.target_damping_coef, self.target_spring_coef = (
            action[-2],
            action[-1],
        )
        self.obstacle_damping_coef = action[:-2:2]
        self.obstacle_spring_coef = action[1:-1:2]

        target_force = (
            self.target_spring_coef * (self.target_pos - self.robot_pos)
            + self.target_damping_coef * (0 - self.robot_vel)
        )
        self.obstacle_vel = np.random.normal(loc=0.0, scale=1.0, size=(self.num_obs, 2))
        self.target_vel = np.random.normal(loc=0, scale=1.5, size=(2))
        
        obstacle_force = np.zeros(2)
        for i, (pos, vel, spring_coef, damping_coef) in enumerate(
            zip(self.obstacle_pos, self.obstacle_vel, self.obstacle_spring_coef, self.obstacle_damping_coef)
            ):
            obs_dist = self.robot_pos - pos
            dist_norm = np.linalg.norm(obs_dist)

            if dist_norm < self.detection_range:
                desired_pos = pos + self.detection_range * obs_dist / dist_norm
                desired_vel = np.zeros_like(vel)
                self.obs_force = (
                    spring_coef * (desired_pos - self.robot_pos) + damping_coef * (desired_vel - self.robot_vel)
                )
                obstacle_force += self.obs_force

        total_force = target_force + obstacle_force
        # total_force = target_force
        # total_force = obstacle_force
        acc = np.clip(total_force /self.robot_size, -self.max_acc, self.max_acc)
        self.robot_vel += acc * self.time_step
        self.robot_vel = np.clip(self.robot_vel, -self.max_speed, self.max_speed)
        self.robot_pos += self.robot_vel * self.time_step
        self.robot_pos = np.clip(self.robot_pos, 0, self.max_distance - self.robot_size)

        self.obstacle_pos += self.obstacle_vel * self.time_step
        self.obstacle_pos = np.clip(self.obstacle_pos, 0, self.max_distance - self.robot_size)
        
        self.target_pos+= self.target_vel * self.time_step
        
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
        return self._get_observation(),reward, done, {} , self.count_in_area
    
    def render(self):
        if not hasattr(self, 'fig') or self.fig.canvas.manager.window is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(-5, self.max_distance+5)
        self.ax.set_ylim(-5, self.max_distance+5)
        
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
                self.ax.text(pos[0] + 0.5, pos[1], f"damping_coef: {self.obstacle_spring_coef[idx]:.2f}\nspring_coef: {self.obstacle_damping_coef[idx]:.2f}", fontsize=8, color='black')
                lines.append([self.robot_pos, pos])
                colors.append(self.obs_force)

        lines.append([self.robot_pos, self.target_pos])
        target_force = np.linalg.norm(self.target_spring_coef * (self.target_pos - self.robot_pos) - self.target_damping_coef * self.robot_vel)


        line_collection = LineCollection(lines, cmap='twilight_shifted_r', linewidths=1)

        self.ax.add_collection(line_collection)
        self.ax.text(self.target_pos[0] + 0.5, self.target_pos[1], f"damping_coef: {self.target_damping_coef:.2f}\nspring_coef: {self.target_spring_coef:.2f}", fontsize=8, color='red')

        # Draw legend
        self.ax.legend()

        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



    def close(self):
        pass