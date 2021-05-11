import argparse
import collections
import os

import gym
import envs
#from gym import wrappers
import numpy as np
import math
import matplotlib.pyplot as plt

from model import Qnetwork
from policy import EpsilonGreedyPolicy

#def obstacle_map(obs_x_lc, obs_y_lc):
#    img_w, img_h = 20, 20
#    img = np.zeros((img_h, img_w))
#
#    x = 0
#    y = 10
#    th = math.radians(0)
#    obs_x_lc = obs_x_lc*1000
#    obs_y_lc = obs_y_lc*-1000
#    z = math.sqrt((obs_x_lc-x)**2+(obs_y_lc-y)**2)/100
#    direction_obs = math.atan2(obs_y_lc, obs_x_lc)
#    a = 1
#
#    for i in range(img_h):
#        inv_i = img_h - i 
#        for j in range(img_w):
#            cx, cy = inv_i-0.5, j+0.5
#            r = math.sqrt((cx-x)**2+(cy-y)**2)
#            yaw = math.atan2(cy-y, cx-x) - th
#
#            if abs(cx-obs_x_lc/100) < a/2 and abs(cy-y-obs_y_lc/100) < a/2:
#                img[i][j] = 1
#            else:
#                img[i][j] = 0
#            #if r > z+a/2 or abs(yaw-direction_obs) > math.atan2(a/2, z):
#            #    img[i][j] = 0
#            #elif abs(r-z) < a/2:
#            #    img[i][j] = 1
#            #elif r <= z:
#            #    img[i][j] = 0
#    ax.imshow(img, cmap="Greys")
##    ax.plot(img)
##    plt.draw()
#    plt.pause(.001)

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_path', help='learned model_weight path')
    args = parser.parse_args()
    weight_path = args.weight_path
    max_step = 250  # 1エピソードの最大ステップ数
    actions_list = [0, 1, 2, 3, 4, 5]  # 行動(action)の取りうる値のリスト
    num_episode = 10
    base_dir = os.path.dirname(weight_path)
    predict_movie_dir = os.path.join(base_dir, 'movie')

#    os.makedirs(predict_movie_dir, exist_ok=True)

    env = gym.make("GankenKun_map_obs-v0")
#    env = wrappers.Monitor(env, predict_movie_dir, force=True, video_callable=(lambda ep: ep % 1 == 0))
#    dim_state = env.env.observation_space.shape[0]
    dim_state = 406

    q_network = Qnetwork(dim_state, actions_list)
    q_network.main_network.load_weights(weight_path)
    policy = EpsilonGreedyPolicy(q_network, epsilon=0)

    exps = collections.namedtuple("exps", ["state", "action", "reward"])
    last_10_totalrewards = np.zeros(10)
    print('start_episodes')
    for episode_step in range(1, num_episode + 1):
        state = env.reset()
        episode_history = []
        score = 0
        step = 0
        while True:
#            env.render()
            step += 1
            action, epsilon, q_values = policy.get_action(state, actions_list)
            next_state, reward, done, info = env.step(action)

            print(step)

#            obs_x_lc, obs_y_lc = state[6], state[7]
#            obstacle_map(obs_x_lc, obs_y_lc)

#            if reward > -1:
#                reward = 1
#            else:
#                reward = -1
            score += reward
            episode_history.append(exps(state=state,
                                        action=action,
                                        reward=reward))
            state = next_state
            if step > max_step or done:
                print('episode_{}, score_{}'.format(episode_step, score))
                print('{}step'.format(step))
                break

        total_reward = sum(e.reward for e in episode_history)
        last_10_totalrewards[episode_step % 10] = total_reward
        last_10_avg = sum(last_10_totalrewards) / (episode_step if episode_step < 10 else 10)
        if episode_step % 10 == 0:
            print('episode_{} score_{}  last 10_{}'.format(episode_step, score, last_10_avg))
#    env.close()


if __name__ == '__main__':
#    fig, ax = plt.subplots()
    predict()
