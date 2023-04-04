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
        
        
        # robot has sensors to detect distance to obstacles
        self.detecting_sensor_number = 16
        self.sensor_angle = 2*np.pi*np.arange(self.detecting_sensor_number)/self.detecting_sensor_number

        self.action_space = gym.spaces.Box(
            low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(    # observation space is sensornum(distance_sensor) + 2(robot_position) + 2(target_position)
            low=-self.max_distance, high=self.max_distance, shape=(self.detecting_sensor_number+2+2+4,), dtype=np.float32  # observation space is 12
        )
        self.reset()


    def reset(self):
        self.rot_seed = 2*np.pi*np.random.uniform()
        self.robot_pos = np.random.uniform(low=self.robot_size, high=self.max_distance - self.robot_size, size=2)
        self.obstacle_pos = np.random.uniform(low=2,
                                              high=self.max_distance - 2,
                                              size=(self.num_obs, 2),
                                              )
        self.obstacle_vel = np.random.normal(loc=0.0, scale=0.6, size=(self.num_obs, 2))
        self.robot_vel = np.zeros(2)
        self.t = 0
        self.count_in_area = 0
        self.target_area = 4*self.robot_size
        self.sensor_signals = np.zeros(self.detecting_sensor_number)

        return self._get_observation()
    
    def _get_observation(self):
        # observation is sensor signal + robot position + target position+ distace to 4 wall
        # robot position is robot position
        # target position is target position
        # sensor signal is distance to obstacles
        # sensor is line sensor that can detect distance to obstacles
        # sensor line's angle is 360/sensor_number
        # sensor line's length is detecting_range
        # sensor line's origin is robot position
        # sensor line's direction is robot orientation
             
        observation = np.zeros(self.detecting_sensor_number+2+2+4)
        observation[-2:] = self.robot_pos
        observation[-4:-2] = self.target_pos
        # distance to 4 wall
        observation[-8:-4] = np.array([self.robot_pos[0],self.robot_pos[1],self.max_distance-self.robot_pos[0],self.max_distance-self.robot_pos[1]]) 
        
        sensor_signals = self.detecting_sensor(self.robot_pos,self.obstacle_pos)
        # print("====sensor_signals====",sensor_signals)
        
        observation[:self.detecting_sensor_number] = sensor_signals
        
        
        
        # for reward
        self.target_dist = self.target_pos - self.robot_pos
        obs_pos = self.obstacle_pos - self.robot_pos
        dists = np.linalg.norm(obs_pos, axis=1)
        self.count_in_area = np.count_nonzero(dists < self.detection_range)
        
        
        
        return observation,self.count_in_area
        

    def detecting_sensor(self,robot_pos,obstacle_pos):
        sensor_signals = np.zeros(self.detecting_sensor_number)
        for i in range(self.detecting_sensor_number):
            sensor_angle = 2*np.pi*i/self.detecting_sensor_number
            sensor_direction = np.array([np.cos(sensor_angle),np.sin(sensor_angle)])
            sensor_origin = robot_pos
            sensor_end = sensor_origin + self.detection_range*sensor_direction
            sensor_line = np.array([sensor_origin,sensor_end])
            sensor_signals[i] = self.detecting_sensor_line(sensor_line,obstacle_pos)
        return sensor_signals
    
    def detecting_sensor_line(self, sensor_line, obstacle_pos):
        sensor_origin = sensor_line[0]
        sensor_end = sensor_line[1]
        sensor_direction = sensor_end - sensor_origin
        sensor_length = np.linalg.norm(sensor_direction)
        sensor_direction = sensor_direction / sensor_length

        min_distance = sensor_length
        for obs_pos in obstacle_pos:
            obs_to_origin = obs_pos - sensor_origin
            projection_length = np.dot(obs_to_origin, sensor_direction)
            if 0 <= projection_length <= sensor_length:
                projected_point = sensor_origin + projection_length * sensor_direction
                distance_to_obstacle = np.linalg.norm(obs_pos - projected_point)
                if distance_to_obstacle < min_distance:
                    min_distance = distance_to_obstacle

        return min_distance

    
            
    def step(self, action):
        self.t += 1
        self.robot_vel = np.clip(action, -self.max_speed, self.max_speed)

        self.obstacle_vel = self._get_obstacle_vel()
        self.robot_vel = self._get_robot_vel()


        # robot position update with velocity, cannot go outside the wall
        self.robot_pos += self.robot_vel * self.time_step
        self.robot_pos = np.clip(self.robot_pos, 0, self.max_distance)

        # obstacle position update with velocity,cannot go outside the wall
        # wall set 0~self.max_distance
        self.obstacle_pos += self.obstacle_vel * self.time_step
        self.obstacle_pos = np.clip(self.obstacle_pos, 0, self.max_distance)

        
        # target position is moving in a circle with radius 7 centered at (10,10)
        self.target_pos = np.float32([10 + 7*np.cos(self.rot_seed + self.t/50), 10 + 7*np.sin(self.rot_seed + self.t/50)])



        observation, count_in_area = self._get_observation()
        reward = self._get_reward(count_in_area)
        done = self._get_done(reward)
        info = {}
        return observation, reward, done, info




    def _get_reward(self, count_in_area):
        reward = 0
        # if robot is far from the target, it gets negative reward depending on the distance
        self.target_dist = self.target_pos - self.robot_pos
        reward -= np.linalg.norm(self.target_dist) / 10

        # if robot position is in target area it gets positive reward
        # reward get bigger as the robot gets closer to the target position
        if np.linalg.norm(self.target_dist) < self.target_area:
            reward += 3*(10- np.linalg.norm(self.target_dist))
            

        # if robot coliide to obstacle, it get big negative reward
        for i in range(self.num_obs):
            if np.linalg.norm(self.robot_pos - self.obstacle_pos[i]) < self.robot_size:
                reward -= 100
        # if robot collide to wall, it gets negative reward
        if self.robot_pos[0] > self.max_distance - self.robot_size or self.robot_pos[0] < self.robot_size:
            reward -= 100
        if self.robot_pos[1] > self.max_distance - self.robot_size or self.robot_pos[1] < self.robot_size:
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
        if self.robot_pos[0] > self.max_distance - self.robot_size or self.robot_pos[0] < self.robot_size:
            done = True
        if self.robot_pos[1] > self.max_distance - self.robot_size or self.robot_pos[1] < self.robot_size:
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
        ax.scatter(self.target_pos[0], self.target_pos[1], c='#8c564b', marker='x', s=200, label='Target')
        
        # draw obstacles
        for i in range(self.num_obs):
            ax.add_patch(Circle(self.obstacle_pos[i], self.robot_size, color='g',label='obstacle'))

        # draw a wall of dotted lines
        ax.plot([0, 0], [self.max_distance, 0], 'k--')
        ax.plot([0, self.max_distance], [self.max_distance, self.max_distance], 'k--')
        ax.plot([self.max_distance, self.max_distance], [self.max_distance, 0], 'k--')
        ax.plot([self.max_distance, 0], [0, 0], 'k--')




        # draw target area
        ax.add_patch(Circle(self.target_pos, self.target_area,fill=True, linestyle='dashed', facecolor=(0.2, 0.1, 0.1, 0.05), zorder=-1))

        # draw detection range
        ax.add_patch(Circle(self.robot_pos, self.detection_range, fill=True, linestyle='dashed', facecolor=(0.1, 0.2, 0.1, 0.1), zorder=-1))

        # draw robot velocity
        ax.quiver(self.robot_pos[0], self.robot_pos[1], self.robot_vel[0], self.robot_vel[1], color='b', scale=10)

        # draw timestep
        ax.text(-self.max_distance, self.max_distance, 'timestep: {}'.format(self.t), fontsize=10)

        # Draw the legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')


        # draw sensor lines that detect obstacles. line draw regardless of obstacle's existence
        for i in range(self.detecting_sensor_number):
            sensor_angle = self.sensor_angle[i]
            sensor_pos = self.robot_pos + self.detection_range * np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
            ax.plot([self.robot_pos[0], sensor_pos[0]], [self.robot_pos[1], sensor_pos[1]], 'g--', linewidth=0.5)
        
        
        # draw the line between robot and target
        ax.plot([self.robot_pos[0], self.target_pos[0]], [self.robot_pos[1], self.target_pos[1]], 'r--', linewidth=0.5)

        plt.pause(0.01)

    def _get_target_force(self):
        target_dist = self.target_dist
        target_force = -self.target_spring_coef * target_dist - self.target_damping_coef * self.robot_vel
        return target_force
    

    
    def close(self):
        pass