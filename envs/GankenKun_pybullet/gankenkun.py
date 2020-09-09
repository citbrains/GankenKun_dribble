#!/usr/bin/env python3

import pybullet as p
import numpy as np
import sys
sys.path.append('./GankenKun')
from kinematics import *
from foot_step_planner import *
from preview_control import *
from walking import *
from random import random, choice 
from time import sleep
import csv

def GankenKun():
    TIME_STEP = 0.001
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(TIME_STEP)
    
    # planeId = p.loadURDF("URDF/plane.urdf", [0, 0, 0])
    FieldId = p.loadSDF("SDF/field/soccerfield.sdf")
    RobotId = p.loadURDF("URDF/gankenkun.urdf", [0, 0, 0])
    BallId = p.loadSDF("SDF/ball/ball.sdf")
    
    index = {p.getBodyInfo(RobotId)[0].decode('UTF-8'):-1,}
    for id in range(p.getNumJoints(RobotId)):
        index[p.getJointInfo(RobotId, id)[12].decode('UTF-8')] = id
    
    left_foot0  = p.getLinkState(RobotId, index[ 'left_foot_link'])[0]
    right_foot0 = p.getLinkState(RobotId, index['right_foot_link'])[0]
    
    joint_angles = []
    for id in range(p.getNumJoints(RobotId)):
        if p.getJointInfo(RobotId, id)[3] > -1:
            joint_angles += [0,]
    
    left_foot  = [ left_foot0[0]-0.015,  left_foot0[1]+0.01,  left_foot0[2]+0.02]
    right_foot = [right_foot0[0]-0.015, right_foot0[1]-0.01, right_foot0[2]+0.02]
    
    pc = preview_control(0.01, 1.0, 0.27)
    walk = walking(RobotId, left_foot, right_foot, joint_angles, pc)
    
    index_dof = {p.getBodyInfo(RobotId)[0].decode('UTF-8'):-1,}
    for id in range(p.getNumJoints(RobotId)):
        index_dof[p.getJointInfo(RobotId, id)[12].decode('UTF-8')] = p.getJointInfo(RobotId, id)[3] - 7
    
    #goal position (x, y) theta
    x_goal, y_goal, th_goal = 0.1, 0.0, 0.0
    foot_step = walk.setGoalPos([x_goal, y_goal, th_goal])

#    print(p.getBasePositionAndOrientation(BallId[0]))
    
    j = 0
    step = 0
    while p.isConnected():
        j += 1
        if j >= 10:
            joint_angles,lf,rf,xp,n = walk.getNextPos()
            j = 0
            if n == 0:
                if (len(foot_step) <= 6):
                    print("robot pose: [{}]".format(p.getBasePositionAndOrientation(RobotId)[0]))
                    x, y, th = p.getBasePositionAndOrientation(RobotId)[0]
            #       x   _goal, y_goal, th = random()-0.5, random()-0.5, random()-0.5
#                    actions_list = [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.06, 0.0], [0.0, -0.06, 0.0], [0.0, 0.0, 0.4], [0.0, 0.0, -0.4]]
                    actions_list = [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]
                    action = choice(actions_list)
                    action = actions_list[0]
                    x_goal = x + action[0]
                    y_goal = y + action[1]
                    th_goal = th + action[2]
            #       x_goal = x + action[0]
            #       y_goal = y + action[1]
            #       th_goal = th + action[2]
                    print("Goal: ("+str(x_goal)+", "+str(y_goal)+", "+str(th_goal)+")")
            #       print(p.getBasePositionAndOrientation(BallId[0])[0])
                    ball_x, ball_y, _ = p.getBasePositionAndOrientation(BallId[0])[0]
                    print("ball pose: [{}, {}]".format(ball_x, ball_y))
                    foot_step = walk.setGoalPos([x_goal, y_goal, th_goal])
                else:
                    foot_step = walk.setGoalPos()
                if step > 10:
                    p.resetBasePositionAndOrientation(RobotId, [0, 0, 0], [0, 0, 0, 1.0])
                    p.resetBasePositionAndOrientation(BallId[0], [0.2, 0, 0.1], [0, 0, 0, 1.0])
                    x_goal, y_goal, th_goal = 0.1, 0.0, 0.0
                    del walk, pc
                    pc = preview_control(0.01, 1.0, 0.27)
                    walk = walking(RobotId, left_foot, right_foot, joint_angles, pc)
                    foot_step = walk.setGoalPos([x_goal, y_goal, th_goal])
                    state = p.getBasePositionAndOrientation(RobotId)[0]
                    print(state)
                    step = 0
                step += 1
           #if you want new goal, please send position
        for id in range(p.getNumJoints(RobotId)):
            qIndex = p.getJointInfo(RobotId, id)[3]
            if qIndex > -1:
                p.setJointMotorControl2(RobotId, id, p.POSITION_CONTROL, joint_angles[qIndex-7])
        
        p.stepSimulation()
    #   sleep(TIME_STEP) # delete -> speed up


if __name__ == '__main__':
    GankenKun()
