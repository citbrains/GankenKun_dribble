#!/usr/bin/env python3

import pybullet as p
import numpy as np
import sys
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


class GankenKunWalkEnv(gym.Env):
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

        TIME_STEP = 0.01
        physicsClient = p.connect(p.GUI)
#        physicsClient = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSubSteps=10)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        self.FieldId = p.loadSDF("envs/GankenKun_pybullet/SDF/field/soccerfield.sdf")
        self.RightPoleId = p.loadSDF("envs/GankenKun_pybullet/SDF/goal/right_pole.sdf")
        self.LeftPoleId = p.loadSDF("envs/GankenKun_pybullet/SDF/goal/left_pole.sdf")
        self.RobotId = p.loadURDF("envs/GankenKun_pybullet/URDF/gankenkun_sub.urdf", [0, 0, 0])
#        self.BallId = p.loadSDF("envs/GankenKun_pybullet/SDF/ball/ball.sdf")
        
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
#        ball_x, ball_y, _ = p.getBasePositionAndOrientation(self.BallId[0])[0]
#        ball_x_lc = (ball_x - x) * math.cos(-yaw) - (ball_y - y) * math.sin(-yaw)
#        ball_y_lc = (ball_x - x) * math.sin(-yaw) + (ball_y - y) * math.cos(-yaw)
#        self.state = [x, y, math.sin(yaw), math.cos(yaw), ball_x_lc, ball_y_lc]
        self.state = [x, y, math.sin(yaw), math.cos(yaw)]
        

#        ball_distance = math.sqrt(ball_x_lc**2 + ball_y_lc**2)
#        ball_direction_deg = math.degrees(math.atan2(ball_y_lc, ball_x_lc))
#        goal_x, goal_y = self.goal_pos
#        ball_goal_distance = math.sqrt((goal_x - ball_x)**2 + (goal_y - ball_y)**2)

        done = bool(
                abs(x) > self.x_threshold
                or abs(y) > self.y_threshold
#                or abs(ball_x) > self.ball_x_threshold
#                or abs(ball_y) > self.ball_y_threshold
#                or ball_distance > self.dist_threshold
#                or abs(ball_direction_deg) > self.direction_deg_threshold
        )
#        reward = 0
#        if not done:
#            reward += math.floor(-10.0 * ball_goal_distance)/10
#        else:
#            if ball_x > goal_x and self.goal_leftpole < ball_y < self.goal_rightpole:
#                reward = 1500
#            elif ball_distance > self.dist_threshold or abs(ball_direction_deg) > self.direction_deg_threshold:
#                reward += math.floor(-200.0 * ball_goal_distance)
#            else:
#                reward = -500
        
#        return self.state, reward, done, {}
        return self.state, done

    def reset(self):
#        init_y = random.uniform(-2.5, 2.5) 
        init_y = 0
        p.resetBasePositionAndOrientation(self.RobotId, [0, init_y, 0], [0, 0, 0, 1.0])
#        p.resetBasePositionAndOrientation(self.BallId[0], [0.2, init_y, 0.1], [0, 0, 0, 1.0])
        p.resetBasePositionAndOrientation(self.RightPoleId[0], [4.5, -1.3, 0.51], [0, 0, 0, 1.0])
        p.resetBasePositionAndOrientation(self.LeftPoleId[0], [4.5, 1.3, 0.51], [0, 0, 0, 1.0])
        x_goal, y_goal, th_goal = 0.0, 0.0, 0.0
        self.foot_step = self.walk.setGoalPos([x_goal, y_goal, th_goal])
        x, y, _ = p.getBasePositionAndOrientation(self.RobotId)[0]
        roll, pitch, yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.RobotId)[1])
        print(p.getBaseVelocity(self.RobotId)[0])
#        ball_x, ball_y, _ = p.getBasePositionAndOrientation(self.BallId[0])[0]
#        ball_x_lc = (ball_x - x) * math.cos(-yaw) - (ball_y - y) * math.sin(-yaw)
#        ball_y_lc = (ball_x - x) * math.sin(-yaw) + (ball_y - y) * math.cos(-yaw)
#        self.state = [x, y, math.sin(yaw), math.cos(yaw), ball_x_lc, ball_y_lc]
        self.state = [x, y, math.sin(yaw), math.cos(yaw)]
        return np.array(self.state)


