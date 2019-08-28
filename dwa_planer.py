'''
给出当前机器人的：观测状态、目标点信息、机器人状态
计算出给定时间内的预测轨迹、目标代价、障碍物代价等，选择最合适的动作

TODO 需要有一个策略：
- 当距离障碍物较近时（被多个障碍物困住时，适当降低）
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import pygame
from pygame_car_env import CarEnv, liner_velocity, angular_velocity

#Config information
MIN_SPEED = 0
MAX_SPEED = 40
MAX_YAWRATE = 40 /180 * np.pi
MAX_ACCEL = 20
DT = 0.1
MAX_DYAWRATE = 100 / 180 * np.pi
V_RESO = 0.1
YAWRATE_RESO = 1 / 180 * np.pi
PREDICT_TIME = 1.5

GOAL_COST_FACTOR = 1
SPEED_COST_FACTOR = 20

def calc_dynamic_window(robot_state):
    '''
    ### 位置空间集合  
    - param: robot_state:[x, y, r, liner, angular]
    '''
    #最大最小速度
    vs = [MIN_SPEED, MAX_SPEED, -MAX_YAWRATE, MAX_YAWRATE]
    # 单个采样周期内能够达到的最大最小速度
    vd = [robot_state[3] - MAX_ACCEL * DT,
          robot_state[3] + MAX_ACCEL * DT,
          robot_state[4] - MAX_DYAWRATE * DT,
          robot_state[4] + MAX_DYAWRATE * DT]
    #求出交集
    vr = [max(vs[0], vd[0]), min(vs[1], vd[1]),
          max(vs[2], vd[2]), min(vs[3], vd[3])]
    return vr
def motion(state, vel):
    '''
    更新位置空间
    ---
    - param state: 机器人状态信息 [posx, posy, r, liner, angular, collision_radius]
    - param vel:
    ---
    - return : updated state
    '''
    state_ = np.array(state)
    state_[0] += vel[0] * np.cos(state[2]) * DT
    state_[1] += vel[0] * np.sin(state[2]) * DT
    state_[2] += vel[1] * DT
    state_[3] = vel[0]
    state_[4] = vel[1]
    return state_

def calc_trajectory(state_, v, w):
    '''
    预测指定时间的轨迹
    '''
    state = np.array(state_)
    trajectory = np.array(state)
    time = 0
    while time <= PREDICT_TIME:
        state = motion(state, [v, w])
        trajectory = np.vstack((trajectory, state))
        time += DT
    return trajectory

def calc_final_input(state, vel, vr, goal, obs):
    '''
    计算采样空间的评价， 选择最合适的最终输入
    ---
    - param state: [pos_x, pos_y, r, liner, angular, coll_radius]
    - param state: [liner, angular]
    - param vr: dynamic window
    - param obs: obs informaiton
    '''
    state_ = state[:]
    min_cost = float('inf')
    best_vel = vel

    best_trajectory = np.array([state])
    for i in range(len(liner_velocity)):
        for j in range(len(angular_velocity)):
            trajectory = calc_trajectory(state_, liner_velocity[i], angular_velocity[j])
            goal_cost = calc_goal_cost(trajectory, goal)
            speed_cost = SPEED_COST_FACTOR * (MAX_SPEED - trajectory[-1,3])
            obs_cost = calc_obs_cost(trajectory, obs, state[5])
            final_cost = goal_cost + speed_cost + obs_cost
            if min_cost >= final_cost:
                min_cost = final_cost
                best_vel = [i, j]
                best_trajectory = trajectory
    return best_vel, best_trajectory
            
def calc_obs_cost(traj, obs, robots_coll_radius):
    '''
    计算到各个障碍物的最短距离，得到obs_cost
    ---
    - param traj: 预测轨迹  
    - param obs: 环境内的全部障碍物，其中各个item的值为[obs_pos_x,obs_pos_y,obs_collision_th]
    - param robot_coll_radius: 机器人的碰撞半径
    - 不要忽略机器人的半径
    '''
    min_r = float('inf')

    for ii in range(len(traj)): # for all point in traj
        for i in range(len(obs)): # for all obs
            # obs position
            ox = obs[i][0]
            oy = obs[i][1]
            # the dictance of robot and obs(min)
            dx = traj[ii, 0] - ox
            dy = traj[ii, 1] - oy
            r = np.sqrt(dx ** 2 + dy ** 2)

            if r <= np.abs(robots_coll_radius + obs[i][2]):
                '''
                两个圆心之间的距离小于各自半径之和，即发生碰撞
                TODO 是否需要将碰撞的代价设置为大数（而非无限大）？
                '''
                return float('inf')
            # update min_r
            if min_r >= r:
                min_r = r
    return 600 - min_r

def calc_goal_cost(traj, goal):
    '''
    计算预测轨N迹到目标点的代价， 距离越大， 代价越大
    ---
    - param traj:
    - param goal:  

    NOTE 是否需要通过图片的形式进行计算
    ---
    - return:
    '''
    dx = goal[0] - traj[-1, 0]
    dy = goal[1] - traj[-1, 1]
    goal_dist = np.sqrt(dx ** 2 + dy ** 2)
    cost = GOAL_COST_FACTOR * goal_dist
    return cost

def dwa_control(robot_state, robot_vel, robot_goal, obscale):
    '''
    dwa算法控制
    '''
    vr = calc_dynamic_window(robot_state)
    robot_vel, trajectory = calc_final_input(
        robot_state, 
        robot_vel, 
        vr, 
        robot_goal, 
        obscale
    )
    return robot_vel, trajectory

def test():
    '''
    独立测试与环境之间的接口用
    '''
    env = CarEnv()
    state, goal = env.agents[0].reset()
    obs = env.obs_info
    vel = np.array([state[3],state[4]])
    trajectory = np.array(state)
    for i in range(1000):
        vel, best_trajectory = dwa_control(state, vel, goal, obs)
        state = env.agents[0].step(vel)#修改step函数，以适应dwa的连续动作输入
        trajectory = np.vstack((trajectory ,state))

        if math.sqrt((state[0]-goal[0])**2 + (state[1]-goal[1])**2) <= state[5]:
            print('Goal!')
            break
if __name__ == '__main__':
    test()