#!/usr/bin/env python3

import pybullet as p
import numpy as np
import sys
import matplotlib.pyplot as plt
#sys.path.append('./GankenKun')
from envs.GankenKun_pybullet.GankenKun.kinematics import *
from envs.GankenKun_pybullet.GankenKun.foot_step_planner import *
from envs.GankenKun_pybullet.GankenKun.preview_control import *
from envs.GankenKun_pybullet.GankenKun.walking import *
import random
from time import sleep
import csv

import gym
from gym import error, spaces, utils
from gym.utils import seeding

fig, ax = plt.subplots()
def obstacle_map(obs_x_lc, obs_y_lc):
    img_w, img_h = 20, 20
    img = np.zeros((img_h, img_w))

    x = 0
    y = 10
    th = math.radians(0)
    obs_x_lc = obs_x_lc*1000
    obs_y_lc = obs_y_lc*-1000
    z = math.sqrt((obs_x_lc-x)**2+(obs_y_lc-y)**2)/100
    direction_obs = math.atan2(obs_y_lc, obs_x_lc)
    a = 1

    for i in range(img_h):
        inv_i = img_h - i 
        for j in range(img_w):
            cx, cy = inv_i-0.5, j+0.5
            r = math.sqrt((cx-x)**2+(cy-y)**2)
            yaw = math.atan2(cy-y, cx-x) - th

            if abs(cx-obs_x_lc/100) < a/2 and abs(cy-y-obs_y_lc/100) < a/2:
                img[i][j] = 1
            else:
                img[i][j] = 0
#            if r > z+a/2 or abs(yaw-direction_obs) > math.atan2(a/2, z):
#                img[i][j] = 0
#            elif abs(r-z) < a/2:
#                img[i][j] = 1
#            elif r <= z:
#                img[i][j] = 0

    ax.imshow(img, cmap="Greys")
    plt.pause(.001)

    return img

class GankenKunObstacleEnv(gym.Env):
    def __init__(self):
        self.state = None
        self.goal_pos = [4.5, 0]
        self.goal_rightpole = self.goal_pos[1] + 1.3
        self.goal_leftpole = self.goal_pos[1] - 1.3
        self.x_threshold = 5.0
        self.y_threshold = 3.5
        self.ball_x_threshold = 4.5
        self.ball_y_threshold = 3.0
        self.direction_deg_threshold = 90
        self.dist_threshold = 0.55
        self.obs_dist_threshold = 0.3
        self.obs_direction_deg_threshold = 45

        TIME_STEP = 0.01
        physicsClient = p.connect(p.GUI)
#        physicsClient = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSubSteps=10)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

#        planeId = p.loadURDF("envs/GankenKun_pybullet/URDF/plane.urdf", [0, 0, 0])
        self.FieldId = p.loadSDF("envs/GankenKun_pybullet/SDF/field/soccerfield.sdf")
        self.RightPoleId = p.loadSDF("envs/GankenKun_pybullet/SDF/goal/right_pole.sdf")
        self.LeftPoleId = p.loadSDF("envs/GankenKun_pybullet/SDF/goal/left_pole.sdf")
        self.RobotId = p.loadURDF("envs/GankenKun_pybullet/URDF/gankenkun_sub.urdf", [0, 0, 0])
        self.BallId = p.loadSDF("envs/GankenKun_pybullet/SDF/ball/ball.sdf")
#        self.ObstacleId_1 = p.loadSDF("envs/GankenKun_pybullet/SDF/obstacle/obstacle_box.sdf")
#        self.ObstacleId_2 = p.loadSDF("envs/GankenKun_pybullet/SDF/obstacle/obstacle_box.sdf")
#        self.ObstacleId_3 = p.loadSDF("envs/GankenKun_pybullet/SDF/obstacle/obstacle_box.sdf")
        self.ObstacleId_4 = p.loadSDF("envs/GankenKun_pybullet/SDF/obstacle/obstacle_box.sdf")
        
        self.index = {p.getBodyInfo(self.RobotId)[0].decode('UTF-8'):-1,}
        for id in range(p.getNumJoints(self.RobotId)):
            self.index[p.getJointInfo(self.RobotId, id)[12].decode('UTF-8')] = id
        
        self.left_foot0  = p.getLinkState(self.RobotId, self.index[ 'left_foot_link'])[0]
        self.right_foot0 = p.getLinkState(self.RobotId, self.index['right_foot_link'])[0]
        
        self.joint_angles = []
        for id in range(p.getNumJoints(self.RobotId)):
            if p.getJointInfo(self.RobotId, id)[3] > -1:
                self.joint_angles += [0,]
        
        self.left_foot  = [ self.left_foot0[0]-0.015,  self.left_foot0[1]+0.01,  self.left_foot0[2]+0.02]
        self.right_foot = [self.right_foot0[0]-0.015, self.right_foot0[1]-0.01, self.right_foot0[2]+0.02]
        
        self.pc = preview_control(0.01, 1.0, 0.27)
        self.walk = walking(self.RobotId, self.left_foot, self.right_foot, self.joint_angles, self.pc)
        
        self.index_dof = {p.getBodyInfo(self.RobotId)[0].decode('UTF-8'):-1,}
        for id in range(p.getNumJoints(self.RobotId)):
            self.index_dof[p.getJointInfo(self.RobotId, id)[12].decode('UTF-8')] = p.getJointInfo(self.RobotId, id)[3] - 7
        
        self.actions_list = [[ self.walk.max_stride_x, 0.0, 0.0],
                             [-self.walk.max_stride_x, 0.0, 0.0],
                             [0.0,  self.walk.max_stride_y, 0.0],
                             [0.0, -self.walk.max_stride_y, 0.0],
                             [0.0, 0.0,  self.walk.max_stride_th],
                             [0.0, 0.0, -self.walk.max_stride_th],]

    def step(self, action_num):
        action = self.actions_list[action_num]
        x_goal = self.foot_step[0][1] + action[0]
        y_goal = self.foot_step[0][2] -self.foot_step[0][5] + action[1]
        th_goal = self.foot_step[0][3] + action[2]    
        self.foot_step = self.walk.setGoalPos([x_goal, y_goal, th_goal])
        while p.isConnected():
            self.joint_angles,lf,rf,xp,n = self.walk.getNextPos()
            if n == 0:
                if self.foot_step[0][4] == 'left':
                    break
                else:
                    self.foot_step = self.walk.setGoalPos()
            #if you want new goal, please send position
            for id in range(p.getNumJoints(self.RobotId)):
                qIndex = p.getJointInfo(self.RobotId, id)[3]
                if qIndex > -1:
                    p.setJointMotorControl2(self.RobotId, id, p.POSITION_CONTROL, self.joint_angles[qIndex-7])
                
            p.stepSimulation()
            #   sleep(TIME_STEP) # delete -> speed up
        x, y, _ = p.getBasePositionAndOrientation(self.RobotId)[0]
        roll, pitch, yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.RobotId)[1])
        ball_x, ball_y, _ = p.getBasePositionAndOrientation(self.BallId[0])[0]
        ball_x_lc = (ball_x - x) * math.cos(-yaw) - (ball_y - y) * math.sin(-yaw)
        ball_y_lc = (ball_x - x) * math.sin(-yaw) + (ball_y - y) * math.cos(-yaw)
#        obs1_x, obs1_y, _ = p.getBasePositionAndOrientation(self.ObstacleId_1[0])[0]
#        obs2_x, obs2_y, _ = p.getBasePositionAndOrientation(self.ObstacleId_2[0])[0]
#        obs3_x, obs3_y, _ = p.getBasePositionAndOrientation(self.ObstacleId_3[0])[0]
#        obs4_x, obs4_y, _ = p.getBasePositionAndOrientation(self.ObstacleId_4[0])[0]
#        obs_x = [obs1_x, obs2_x, obs3_x, obs4_x]
#        obs_y = [obs1_y, obs2_y, obs3_y, obs4_y]
#        obs_distance = []
#        for i in range(4):
#            obs_distance.append(math.sqrt((obs_x[i] - x)**2 + (obs_y[i] - y)**2))
#        num = np.argmin(obs_distance)
#        obs_x, obs_y = obs_x[num], obs_y[num]
        obs_x, obs_y, _ = p.getBasePositionAndOrientation(self.ObstacleId_4[0])[0]
        obs_x_lc = (obs_x - x) * math.cos(-yaw) - (obs_y - y) * math.sin(-yaw)
        obs_y_lc = (obs_x - x) * math.sin(-yaw) + (obs_y - y) * math.cos(-yaw)
        
        obs_img = obstacle_map(obs_x_lc, obs_y_lc)
        obs_img = obs_img.reshape(400)
        print(y)
        self.state = np.append([x, y, math.sin(yaw), math.cos(yaw), ball_x_lc, ball_y_lc], obs_img)
        self.state = self.state.tolist()
#        self.state = [x, y, math.sin(yaw), math.cos(yaw), ball_x_lc, ball_y_lc, obs_x_lc, obs_y_lc]
#        self.state = [x, y, math.sin(yaw), math.cos(yaw), ball_x, ball_y]

        ball_distance = math.sqrt(ball_x_lc**2 + ball_y_lc**2)
        ball_direction_deg = math.degrees(math.atan2(ball_y_lc, ball_x_lc))
        goal_x, goal_y = self.goal_pos
        ball_goal_distance = math.sqrt((goal_x - ball_x)**2 + (goal_y - ball_y)**2)
        obs_distance = math.sqrt(obs_x_lc**2 + obs_y_lc**2)
        ball_obs_distance = math.sqrt((obs_x - ball_x)**2 + (obs_y - ball_y)**2)

        done = bool(
                abs(x) > self.x_threshold
                or abs(y) > self.y_threshold
                or abs(ball_x) > self.ball_x_threshold
                or abs(ball_y) > self.ball_y_threshold
                or ball_distance > self.dist_threshold
                or abs(ball_direction_deg) > self.direction_deg_threshold
                or obs_distance < self.obs_dist_threshold
                or ball_obs_distance < self.obs_dist_threshold
        )
        reward = 0
        if not done:
#            reward += math.floor(-10.0 * ball_goal_distance)/10
            dis_ball_goal = round(ball_goal_distance, 1)
            dis_ball_obs =round(ball_obs_distance, 1)
            ball_obs_threshold = 0.5 
            ball_goal_threshold = 4.5
            
            if 0 <= dis_ball_goal <= ball_goal_threshold:
                ball_goal_reward = 1 - 2 * (dis_ball_goal / ball_goal_threshold)
            elif ball_goal_threshold < dis_ball_goal:
                ball_goal_reward = -1
            
            if 0 <= dis_ball_obs <= ball_obs_threshold:
                ball_obs_reward = 2 * (dis_ball_obs / ball_obs_threshold) - 1
            elif ball_obs_threshold < dis_ball_obs:
                ball_obs_reward = 1
            
            w1 = 0.8
            w2 = 1 - w1
            reward = round(w1*ball_goal_reward + w2*ball_obs_reward, 2) 

        else:
            if ball_x > goal_x and self.goal_leftpole < ball_y < self.goal_rightpole:
                reward = 1500
            elif ball_distance > self.dist_threshold or abs(ball_direction_deg) > self.direction_deg_threshold:
                reward = math.floor(-100.0 * ball_goal_distance)
            else:
                reward = -500
        
        return self.state, reward, done, {}

    def reset(self):
#        init_y = random.uniform(-2.5, 2.5) 
        init_y = 0

#        init_ball_x = random.uniform(0.2, 0.4)
#        init_ball_y = init_y + random.uniform(-0.1, 0.1)
        init_ball_x = 0.2
        init_ball_y = 0.0

#        obs_x = []
#        obs_y = []
#        arr_x = np.arange(1, 4.3, 0.3)
#        list_x = arr_x.tolist()
#        obs_x = random.sample(list_x, 3)
#        obs_y = [random.randint(-2500, 2500)/1000 for i in range(3)]
#        obs_x.append(random.randint(4100, 4500)/1000)
#        obs_y.append(random.randint(-1000, 1000)/1000)
#
#        obs_x = [1.5, 3.0, 3.5, 4.5]
#        obs_y = [0.0, 0.5, -1.0, -0.3]

        obs_x = 1.0
        obs_y = 0.0
#        obs_x = random.randint(1000, 4500)/1000
#        obs_y = random.randint(-500, 500)/1000

        p.resetBasePositionAndOrientation(self.RobotId, [0, init_y, 0], [0, 0, 0, 1.0])
        p.resetBasePositionAndOrientation(self.BallId[0], [init_ball_x, init_ball_y, 0.1], [0, 0, 0, 1.0])
        p.resetBasePositionAndOrientation(self.RightPoleId[0], [4.5, -1.3, 0.51], [0, 0, 0, 1.0])
        p.resetBasePositionAndOrientation(self.LeftPoleId[0], [4.5, 1.3, 0.51], [0, 0, 0, 1.0])
#        p.resetBasePositionAndOrientation(self.ObstacleId_1[0], [obs_x[0], obs_y[0], 0.3], [0, 0, 0, 1.0])
 #       p.resetBasePositionAndOrientation(self.ObstacleId_2[0], [obs_x[1], obs_y[1], 0.3], [0, 0, 0, 1.0])
 #       p.resetBasePositionAndOrientation(self.ObstacleId_3[0], [obs_x[2], obs_y[2], 0.3], [0, 0, 0, 1.0])
        p.resetBasePositionAndOrientation(self.ObstacleId_4[0], [obs_x, obs_y, 0.3], [0, 0, 0, 1.0])
        x_goal, y_goal, th_goal = 0.0, 0.0, 0.0
        self.foot_step = self.walk.setGoalPos([x_goal, y_goal, th_goal])
        x, y, _ = p.getBasePositionAndOrientation(self.RobotId)[0]
        roll, pitch, yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.RobotId)[1])
        ball_x, ball_y, _ = p.getBasePositionAndOrientation(self.BallId[0])[0]
        ball_x_lc = (ball_x - x) * math.cos(-yaw) - (ball_y - y) * math.sin(-yaw)
        ball_y_lc = (ball_x - x) * math.sin(-yaw) + (ball_y - y) * math.cos(-yaw)
#        obs_distance = []
#        for i in range(4):
#            obs_distance.append(math.sqrt((obs_x[i] - x)**2 + (obs_y[i] - y)**2))
#        num = np.argmin(obs_distance)
#        obs_x, obs_y = obs_x[num], obs_y[num]
        obs_x_lc = (obs_x - x) * math.cos(-yaw) - (obs_y - y) * math.sin(-yaw)
        obs_y_lc = (obs_x - x) * math.sin(-yaw) + (obs_y - y) * math.cos(-yaw)

        obs_img = obstacle_map(obs_x_lc, obs_y_lc)
        obs_img = obs_img.reshape(400)
        self.state = np.append([x, y, math.sin(yaw), math.cos(yaw), ball_x_lc, ball_y_lc], obs_img)
        self.state = self.state.tolist()
#
#        self.state = [x, y, math.sin(yaw), math.cos(yaw), ball_x_lc, ball_y_lc, obs_x_lc, obs_y_lc]
#        self.state = [x, y, math.sin(yaw), math.cos(yaw), ball_x, ball_y]
        return np.array(self.state)
