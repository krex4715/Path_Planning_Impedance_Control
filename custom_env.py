import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection




class MobileRobotEnv(gym.Env):
    def __init__(self, num_obs=6, maxtime=1000):
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
        self.noise_ratio=0.5


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
        self.obstacle_pos = np.random.uniform(low=2,
                                              high=self.max_distance - 2,
                                              size=(self.num_obs, 2),
                                              )
        self.obstacle_vel = np.random.normal(loc=0.0, scale=1.0, size=(self.num_obs, 2))
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
        self.t += 1
        self.robot_vel = np.clip(action, -self.max_speed, self.max_speed)

        self.obstacle_vel = self._get_obstacle_vel()
        self.robot_vel = self._get_robot_vel()


        # robot position update with velocity, cannot go outside the wall
        self.robot_pos += self.robot_vel * self.time_step
        self.robot_pos = np.clip(self.robot_pos, 0, self.max_distance)

        # obstacle position update with velocity,cannot go outside the wall
        self.obstacle_pos += self.obstacle_vel * self.time_step
        self.obstacle_pos = np.clip(self.obstacle_pos, 0, self.max_distance)

        
        # target position is moving in a circle with radius 7 centered at (10,10)
        self.target_pos = np.float32([10 + 7*np.cos(self.rot_seed + self.t/30), 10 + 7*np.sin(self.rot_seed + self.t/30)])



        observation, count_in_area = self._get_observation()
        reward = self._get_reward(count_in_area)
        done = self._get_done(reward)
        info = {}
        return observation, reward, done, info




    def _get_reward(self, count_in_area):
        reward = 0
        # if robot is far from the target, it gets negative reward depending on the distance
        reward -= np.linalg.norm(self.target_dist) / 10

        # if robot position is in target area it gets positive reward
        # reward get bigger as the robot gets closer to the target position
        if np.linalg.norm(self.target_dist) < self.target_area:
            reward += 10 - np.linalg.norm(self.target_dist)
            

        # if robot coliide to obstacle, it get big negative reward
        for i in range(self.num_obs):
            if np.linalg.norm(self.robot_pos - self.obstacle_pos[i]) < self.robot_size:
                reward -= 100
        # if robot collide to wall, it gets negative reward
        if np.linalg.norm(self.robot_pos) > self.max_distance - self.robot_size:
            reward -= 100
    
        return reward
    
    def _get_done(self, reward):
        done = False


        # if time is over
        if self.t > self.maxtime:
            done = True

        # # if robot position is coinside with target position
        # if np.linalg.norm(self.target_dist) < self.target_area:
        #     done = True


        # if robot collide to obstacle
        for i in range(self.num_obs):
            if np.linalg.norm(self.robot_pos - self.obstacle_pos[i]) < self.robot_size:
                done = True
        # if robot collide to wall
        if np.linalg.norm(self.robot_pos) > self.max_distance -0.2:
            done = True
        elif np.linalg.norm(self.robot_pos) < 0.2:
            done = True

        return done
    
    def _get_robot_vel(self):
        robot_vel = self.robot_vel
        if np.linalg.norm(self.target_dist) < self.target_area:
            return np.zeros(2)
        robot_vel += self._get_target_force() / self.robot_mass * self.time_step
        robot_vel = np.clip(robot_vel, -self.max_speed, self.max_speed)
        return robot_vel
    
    def _get_obstacle_vel(self):
        # obstacle's velocity is updated with sometimes random noise, sometimes circle motion (radius is 1)
        obstacle_vel = self.obstacle_vel
        for i in range(self.num_obs):
            if np.random.uniform() < self.noise_ratio:
                obstacle_vel[i] = np.random.normal(loc=0.0, scale=1.0, size=2)
            else:
                obstacle_vel[i] = self._get_circle_motion(i)
        return obstacle_vel
    
    def _get_circle_motion(self, i):
        # obstacle's velocity is updated with circle motion
        # circle center is the target position raduis is random
        circle_center = self.target_pos
        radius = np.random.uniform(0, 4)
        circle_motion = self.obstacle_pos[i] - circle_center
        circle_motion = np.array([-circle_motion[1], circle_motion[0]])
        circle_motion = circle_motion / np.linalg.norm(circle_motion)
        circle_motion *= radius
        return circle_motion
    

    def _get_target_force(self):
        # target force is calculated with target distance
        target_force = self.target_dist
        target_force = target_force / np.linalg.norm(target_force)
        target_force *= self.max_speed
        return target_force

    
    
    def render(self, mode='human'):
        plt.clf()
        ax = plt.gca()
        ax.set_xlim(-3, self.max_distance+3)
        ax.set_ylim(-3, self.max_distance+3)
        ax.set_aspect('equal')
        ax.add_patch(Circle(self.robot_pos, self.robot_size, color='b',label='robot'))
        ax.add_patch(Circle(self.target_pos, self.robot_size, color='r',label='target'))
        
        # draw obstacles
        for i in range(self.num_obs):
            ax.add_patch(Circle(self.obstacle_pos[i], self.robot_size, color='g',label='obstacle'))

        # draw a wall of dotted lines
        ax.plot([0, 0], [self.max_distance, 0], 'k--')
        ax.plot([0, self.max_distance], [self.max_distance, self.max_distance], 'k--')
        ax.plot([self.max_distance, self.max_distance], [self.max_distance, 0], 'k--')
        ax.plot([self.max_distance, 0], [0, 0], 'k--')




        # draw target area
        ax.add_patch(Circle(self.target_pos, self.target_area, color='r', fill=False))

        # draw detection range
        ax.add_patch(Circle(self.robot_pos, self.detection_range, color='b', fill=False))

        # draw robot velocity
        ax.quiver(self.robot_pos[0], self.robot_pos[1], self.robot_vel[0], self.robot_vel[1], color='b', scale=10)

        # draw timestep
        ax.text(-self.max_distance, self.max_distance, 'timestep: {}'.format(self.t), fontsize=10)

        # Draw the legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        # show the order of obstacles
        for i in range(self.count_in_area):
            ax.text(self.obstacle_pos[self.closest_indices[i]][0], self.obstacle_pos[self.closest_indices[i]][1], str(i), fontsize=10)


        plt.pause(0.01)

    def _get_target_force(self):
        target_dist = self.target_dist
        target_force = -self.target_spring_coef * target_dist - self.target_damping_coef * self.robot_vel
        return target_force
    

    
    def close(self):
        pass