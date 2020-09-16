import os
import random

import gym
import envs
import numpy as np
import matplotlib.pyplot as plt

from model import Qnetwork
from policy import EpsilonGreedyPolicy
from util import now_str, RecordHistory


def train():
    # setup ===========================
    max_episode = 1000  # 学習において繰り返す最大エピソード数
    max_step = 250  # 1エピソードの最大ステップ数
    n_warmup_steps = 10000  # warmupを行うステップ数
    interval = 1  # モデルや結果を吐き出すステップ間隔
    actions_list = [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.06, 0.0], [0.0, -0.06, 0.0], [0.0, 0.0, 0.4], [0.0, 0.0, -0.4]]  # 行動(action)の取りうる値のリスト
    gamma = 0.99  # 割引率
    epsilon = 0.1  # ε-greedyのパラメータ
    memory_size = 10000
    batch_size = 32
    result_dir = os.path.join('./result/dqn', now_str())
    x = []
    x_reward = []
    y_loss = []
    y_td_error = []
    y_reward = []
    y_last10_reward = []

    # インスタンス作成 ==================
    os.makedirs(result_dir, exist_ok=True)
    print(result_dir)
    env = gym.make('GankenKun-v0')
    dim_state = 5
#    print('state:', dim_state)
    q_network = Qnetwork(dim_state, actions_list, gamma=gamma)
    policy = EpsilonGreedyPolicy(q_network, epsilon=epsilon)
    header = ["num_episode", "loss", "td_error", "reward_avg"]
    recorder = RecordHistory(os.path.join(result_dir, "history.csv"), header)
    recorder.generate_csv()

    # warmup=======================
    print('warming up {:,} steps...'.format(n_warmup_steps))
    memory = []
    total_step = 0
    step = 0
    state = env.reset()
    while True:
        step += 1
        total_step += 1

        action = random.choice(actions_list)
        epsilon, q_values = 1.0, None

        next_state, reward, done, info = env.step(action)

        # reward clipping
#        if reward < -1:
#            c_reward = -1
#        else:
#            c_reward = 1
        c_reward = reward
        memory.append((state, action, c_reward, next_state, done))
        state = next_state

        if step > max_step or done:
            state = env.reset()
            step = 0
        if total_step > n_warmup_steps:
            break
    memory = memory[-memory_size:]
    print('warming up {:,} steps... done.'.format(n_warmup_steps))

    # training======================
    print('training {:,} episodes...'.format(max_episode))
    num_episode = 0
    episode_loop = True
    while episode_loop:
        num_episode += 1
        step = 0
        step_loop = True
        episode_reward_list, loss_list, td_list = [], [], []
        state = env.reset()

        while step_loop:
            step += 1
            total_step += 1
            action, epsilon, q_values = policy.get_action(state, actions_list)
            next_state, reward, done, info = env.step(action)

            # reward clipping
#            if reward < -1:
#                c_reward = -1
#            else:
#                c_reward = 1
            c_reward = reward

            memory.append((state, action, c_reward, next_state, done))
            episode_reward_list.append(c_reward)
            exps = random.sample(memory, batch_size)
            loss, td_error = q_network.update_on_batch(exps)
            loss_list.append(loss)
            td_list.append(td_error)

            q_network.sync_target_network(soft=0.01)
            state = next_state
            memory = memory[-memory_size:]

            # end of episode
            if step >= max_step or done:
                step_loop = False
                reward_avg = np.mean(episode_reward_list)
                reward_sum = np.sum(episode_reward_list)
                loss_avg = np.mean(loss_list)
                td_error_avg = np.mean(td_list)
                print("{}episode  reward_avg:{} loss:{} td_error:{}".format(num_episode, reward_sum, loss_avg, td_error_avg))
                if num_episode % interval == 0:
                    model_path = os.path.join(result_dir, 'episode_{}.h5'.format(num_episode))
                    q_network.main_network.save(model_path)
                    history = {
                        "num_episode": num_episode,
                        "loss": loss_avg,
                        "td_error": td_error_avg,
                        "reward_avg": reward_avg
                    }
                    recorder.add_histry(history)
                x.append(num_episode)
                y_loss.append(loss_avg)
                y_td_error.append(td_error_avg)
                y_reward.append(reward_sum)
                if num_episode % 10 == 0:
                    last_10_avg = sum(y_reward[-10:])/len(y_reward[-10:])
                    print("last_10_reward:{}".format(last_10_avg))
                    x_reward.append(num_episode)
                    y_last10_reward.append(last_10_avg)
                

        if num_episode >= max_episode:
            episode_loop = False

#    env.close()
    print('training {:,} episodes... done.'.format(max_episode))
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax1.plot(x, y_loss)
    ax2.plot(x, y_td_error)
    ax3.plot(x, y_last10_reward)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('episode')
    ax2.set_ylabel('td_error')
    ax2.set_xlabel('episode')
    ax3.set_ylabel('reward')
    ax3.set_xlabel('episode')
    plt.show()


if __name__ == '__main__':
    train()
