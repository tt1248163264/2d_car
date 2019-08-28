'''
TODO:
    1.  障碍物随机生成过程中仍会有重叠的现象，暂时找不到原因，
        但是障碍物的重叠一定程度上增加了场景的复杂度，故而利好
    2.  小车的初始位置定为[300, 300]，这样在第一次reset时可能会用透明覆盖掉部分障碍物，不过这个同上一点，会增加场景复杂度，利好
    3.  最终的局部观测图片似乎有一定的抖动现象（随小车朝向的变化）
    4.  各自目标点的设计
    5. 局部观测图片中的目标点
'''

import numpy as np
import random
import pygame
import car_env_utils as env_utils
import time
from pygame.locals import *
from PIL import Image, ImageChops

FPS = 30
SCREEN_WIDTH = int(600)
SCREEN_HEIGHT = int(600)
SCREEN_CENTER = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]

pygame.init()
FPSCLOCK = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
background = (255, 255, 255)
pygame.display.set_caption('Car Env')

IMAGES = env_utils.load()
angular_velocity = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], dtype=np.int32)
liner_velocity = np.array([-40, -30, -20, -10, 0, 10, 20, 30, 40], dtype = np.int32)
dt = 0.1

class CarEnv(object):
    n_sensor = 32
    n_angular = angular_velocity.shape[0]#11
    n_liner = liner_velocity.shape[0]#9
    sensor_max = 1000
    speed_max = 40
    n_obs = len(IMAGES['obs'])#not the number of the obs!!!
    n_agent = 1
    hard_level = 1
    def __init__(self):
        screen.fill(background)
        self._gen_random_obs()
        self.background_image = pygame.surfarray.array3d(screen)
        self.agents = []
        for i in range(self.n_agent):
            agent = Agent(id=i)
            self.agents.append(agent)
            self.agents[i].reset()

    def reset(self, id=None):
        '''
        废弃函数
        '''
        if id is None or id >= self.n_agent:
            return
        car_collision = int(self.agents[id].collision_th)
        current_state = self.get_state()
        x = y = 0
        while True:
            x = random.randint(car_collision, SCREEN_WIDTH-car_collision)
            y = random.randint(car_collision, SCREEN_HEIGHT-car_collision)
            x_start = x - car_collision
            x_end = x + car_collision
            y_start = y - car_collision
            y_end = y + car_collision
            check_area = current_state[y_start:y_end, x_start:x_end,:]
            if (check_area == 255).all():
                break

        #get center point
        self.agents[id].reset(x, y)

    def get_state(self):
        state_img = pygame.surfarray.array3d(screen)
        state_img = state_img.transpose(1, 0, 2)
        return state_img
    
    def clear_noisy(self):
        current_screen = pygame.surfarray.array3d(screen)

    def _gen_random_obs(self):
        '''
        NOTE: hard level: 0 : easy(10), 1:middle(20), 2:hard(30)
        generate base obscales
        '''
        def check_in(avoid_area, x, y, h, w):
            x_ = x + w
            y_ = y + h

            for area in avoid_area:
                x_l = (x >= area[0] and x <= area[2])
                x_r = (x_ >= area[0] and x_ <= area[2])
                y_u = (y >= area[1] and y <= area[3])
                y_d = (y_ >= area[1] and y_ <= area[3])
                if (x_l and y_u) or (x_r and y_u) or (x_l and y_d) or (x_r and y_d):
                    return False
            return True 
        self.obs_info = []
        _obs_index = [10,20,30]
        num_obs = _obs_index[self.hard_level]
        avoid_area = [] # (x,y,x_e,y_e)
        for i in range(num_obs):
            obs = random.choice(IMAGES['obs'])
            height = obs.get_height()
            width = obs.get_width()
            
            while True:
                x = random.randint(1, SCREEN_WIDTH - width)
                y = random.randint(1, SCREEN_HEIGHT - height)
                avoid = check_in(avoid_area, x, y, height, width)
                if avoid:
                    screen.blit(obs, (x, y))
                    radius = max(obs.get_height(),obs.get_width()) / 2
                    self.obs_info.append([x+radius, y+radius, radius])#[x,y,radius]
                    avoid_area.append([x,y,x+width,y+height])
                    break
        pygame.display.update()

class Agent(object):
    def __init__(self, id):
        self.id = id
        self.terminal = False
        self.collision = False
        #TODO:random position!!!
        self.car_state = np.array([600,600,0], dtype = np.float64)
        self.reward = 0

        car_ = random.randint(0, 1)
        self.car_img = IMAGES['car'][car_]
        self.car_mask = IMAGES['mask'][car_]
        self.CAR_WIDTH = self.car_img.get_width()
        self.CAR_HEIGHT = self.car_img.get_height()

        self.collision_th = max(self.CAR_HEIGHT, self.CAR_WIDTH) // 2 + 5

        self.goal_info = np.array([600,600,self.id], dtype = np.int)
        self.goal = IMAGES['goal'][self.id]
        self.goal_mask = IMAGES['goal_mask']

    def step(self, input_action):
        pygame.event.pump()
        
        # normal action
        #liner_vel = input_action[0]
        #angular_vel = input_action[1]
        
        ## discrete action
        liner_vel = liner_velocity[input_action[0]]
        angular_vel = angular_velocity[input_action[1]]

        self.collision = False
        self.reward = 0

        # cover the old car image
        x = int(self.car_state[0])
        y = int(self.car_state[1])
        screen.blit(self.car_mask, (x, y))

        ## left up point update
        c_x = self.car_state[0] + self.CAR_WIDTH / 2
        c_y = self.car_state[1] + self.CAR_HEIGHT / 2
        sita = self.car_state[2] / 180 * np.pi
        c_x += liner_vel * np.cos(sita) * dt
        c_y += liner_vel * np.sin(sita) * dt
        self.car_state[0] = c_x - self.CAR_WIDTH / 2
        self.car_state[1] = c_y - self.CAR_HEIGHT / 2 
        self.car_state[2] += angular_vel / np.pi * 180 * dt
        
        # screen image position update
        x = int(self.car_state[0])
        y = int(self.car_state[1])
        # screen update
        screen.blit(self.car_img, (x, y))
        pygame.display.update()

        dx = int(SCREEN_CENTER[0] - self.car_state[0] - self.CAR_WIDTH // 2)
        dy = int(SCREEN_CENTER[0] - self.car_state[1] - self.CAR_HEIGHT // 2)
        img_array = pygame.surfarray.array3d(screen)
        img_array = img_array.transpose(1,0,2)
        if dx >= 0:
            if dy >= 0:
                img_array[SCREEN_HEIGHT - dy: ,:] = [0,0,0]
                img_array[:SCREEN_HEIGHT-dy, SCREEN_WIDTH - dx:] = [0,0,0]
            else:
                img_array[:-dy,:] = [0,0,0]
                img_array[-dy:,SCREEN_WIDTH-dx:] = [0,0,0]
        else:
            if dy >= 0:
                img_array[SCREEN_HEIGHT - dy:,:] = [0,0,0]
                img_array[:SCREEN_HEIGHT - dy, :-dx] = [0,0,0]
            else:
                img_array[:-dy,:] = [0,0,0]
                img_array[-dy:, :-dx] = [0,0,0]
        row_image = Image.fromarray(img_array)
        offset_image = ImageChops.offset(row_image,dx,dy)
        rotate_image = offset_image.rotate(self.car_state[2]  + 90)
        rotate_array = np.array(rotate_image)
        final_array = rotate_array[SCREEN_CENTER[0]-140:SCREEN_CENTER[0]+20, SCREEN_CENTER[1]-80:SCREEN_CENTER[1]+80, :]

        collision_scale = rotate_array[
            SCREEN_CENTER[0]-self.collision_th:SCREEN_CENTER[0]+self.collision_th, 
            SCREEN_CENTER[1]-self.collision_th:SCREEN_CENTER[1]+self.collision_th, :]
        for x in range(collision_scale.shape[0]):
            i = x - self.collision_th
            for y in range(collision_scale.shape[1]):
                j = y - self.collision_th
                if (i**2 + j**2) <= self.collision_th**2 and (collision_scale[x,y] == [0,0,0]).all():
                    self.collision = True
                    self.reward = -10
        # normal action
        pos_x = self.car_state[0] + self.CAR_WIDTH / 2
        pos_y = self.car_state[1] + self.CAR_HEIGHT / 2
        r = self.car_state[2]  / 180 * np.pi
        state = np.array([pos_x,pos_y, r, liner_vel, angular_vel, self.collision_th])
        return state
        #return self.car_state, final_array, self.reward, self.collision
    
    def reset(self):
        '''
        NOTE:reset agent
        '''
        x = int(self.car_state[0])
        y = int(self.car_state[1])
        screen.blit(self.car_mask, (x, y))

        pos = self.get_space(int(self.collision_th))

        self.car_state[0] = pos[0]
        self.car_state[1] = pos[1]
        self.car_state[2] = 0

        x = int(self.car_state[0])
        y = int(self.car_state[1])
        screen.blit(self.car_img, (x, y))

        self.global_goal()
        pygame.display.update()
        pos_x = self.car_state[0] + self.CAR_WIDTH / 2
        pos_y = self.car_state[1] + self.CAR_HEIGHT / 2
        r = self.car_state[2]
        goal_x = self.goal_info[0] + self.goal.get_width() / 2
        goal_y = self.goal_info[1] + self.goal.get_height() / 2
        return [pos_x, pos_y, r, 0, 0, self.collision_th], [goal_x, goal_y]
    
    def get_space(self, collision_len):
        current_state = self.get_state()
        x_start = y_start = x_end = y_end = 0
        while True:
            x_start = random.randint(1, SCREEN_WIDTH -2*collision_len)
            y_start = random.randint(1, SCREEN_HEIGHT-2*collision_len)
            x_end = x_start + collision_len * 2
            y_end = y_start + collision_len * 2
            check_area = current_state[y_start:y_end, x_start:x_end,:]
            if (check_area == 255).all():
                break

        return [x_start,y_start,x_end,y_end]

    def get_state(self):
        state_img = pygame.surfarray.array3d(screen)
        state_img = state_img.transpose(1, 0, 2)
        return state_img
    
    def global_goal(self):
        #每个机器人使用不同颜色的目标点，以便区分
        x = int(self.goal_info[0])
        y = int(self.goal_info[1])
        screen.blit(self.goal_mask, (x, y))

        collision_th = self.goal.get_size()[0] // 2
        pos = self.get_space(int(collision_th))

        self.goal_info[0] = pos[0]
        self.goal_info[1] = pos[1]

        x = int(self.goal_info[0])
        y = int(self.goal_info[1])
        screen.blit(self.goal, (x, y))


if __name__ == '__main__':
    #import matplotlib.pyplot as plt
    #plt.ion()
    #plt.show()
    env = CarEnv()
    i = 0
    while True:
        state, image,r,c = env.agents[0].step([8,5])
        s_,i_,r_,c_ = env.agents[1].step([8,5])
        #plt.imshow(image)
        #plt.draw()
        #plt.pause(0.001)
        
        if c:
            print('reset  a',i)
            env.agents[0].reset()
            i+=1
        if c_:
            print('reset  b',i)
            env.agents[1].reset()
            i+=1
        time.sleep(0.001)
        