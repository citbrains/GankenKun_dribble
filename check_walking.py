#!/usr/bin/env python3

import gym
import envs
from time import sleep

env = gym.make('GankenKun-v0')
state = env.reset()

step = 0
while True:
    if 0 <= step <= 10:
        action = [1.0, 0.0, 0.0]
    if 10 <= step <= 20:
        action = [0.0, 1.0, 0.0]
    if 20 <= step <= 30:
        action = [-1.0, 0.0, 0.0]
    if 30 <= step <= 40:
        action = [0.0, -1.0, 0.0]
    if 40 <= step <= 50:
        action = [0.0, 0.0, 1.0]
    if 50 <= step <= 60:
        action = [0.0, 0.0, -1.0]
    if 60 <= step <= 70:
        action = [0.0, 0.0, 0.0]
    next_state, reward, done, info = env.step(action)
    step += 1
    sleep(1)

