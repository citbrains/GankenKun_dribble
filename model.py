import numpy as np
from tensorflow.keras.layers import Input, Dense, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from util import idx2mask


class Qnetwork:
    def __init__(self, dim_state, actions_list, gamma=0.99, lr=1e-3, double_mode = True):
        self.dim_state = dim_state
        self.actions_list = actions_list
        self.action_len = len(actions_list)
        self.optimizer = Adam(lr=lr)
        self.gamma = gamma
        self.double_mode = double_mode

        self.main_network = self.build_graph()
        self.target_network = self.build_graph()
        self.trainable_network = self.build_trainable_graph(self.main_network)

    def build_graph(self):
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = self.action_len * 10
        nb_dense_2 = int(np.sqrt(self.action_len * 10 * self.dim_state * 10))

        l_input = Input(shape=(self.dim_state,), name='input_state')
        l_dense_1 = Dense(nb_dense_1, activation='relu', name='hiden_1')(l_input)
        l_dense_2 = Dense(nb_dense_2, activation='relu', name='hiden_2')(l_dense_1)
        l_dense_3 = Dense(nb_dense_3, activation='relu', name='hiden_3')(l_dense_2)
        l_output = Dense(self.action_len, activation='linear', name='output')(l_dense_3)

        model = Model(inputs=[l_input], outputs=[l_output])
        model.summary()
        model.compile(optimizer=self.optimizer, loss='mse')

        return model

    def build_trainable_graph(self, network):
        action_mask_input = Input(shape=(self.action_len,), name='a_mask_inp')
        q_values = network.output
        q_values_taken_action = Dot(axes=-1, name='qs_a')([q_values, action_mask_input])
        trainable_network = Model(inputs=[network.input, action_mask_input], outputs=q_values_taken_action)
        trainable_network.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])

        return trainable_network

    def update_on_batch(self, exps):
        (state, action, reward, next_state, done) = zip(*exps)
        action_index = [self.actions_list.index(a) for a in action]
        action_mask = np.array([idx2mask(a, self.action_len) for a in action_index])
        state = np.array(state)
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array(done)

        next_target_q_values_batch = self.target_network.predict_on_batch(next_state)
        next_q_values_batch = self.main_network.predict_on_batch(next_state)

        if self.double_mode:
            future_return = [next_target_q_values[np.argmax(next_q_values)] for next_target_q_values, next_q_values in zip(next_target_q_values_batch, next_q_values_batch)]

        else:
            future_return = [np.max(naxt_q_values) for next_q_values in next_target_q_values_batch]
        y = reward + self.gamma * (1 - done) * future_return
        loss, td_error = self.trainable_network.train_on_batch([state, action_mask], np.expand_dims(y, -1))

        return loss, td_error

    def sync_target_network(self, soft):
        weights = self.main_network.get_weights()
        target_weights = self.target_network.get_weights()
        for idx, w in enumerate(weights):
            target_weights[idx] *= (1 - soft)
            target_weights[idx] += soft * w
        self.target_network.set_weights(target_weights)
        

#if __name__ == '__main__':
#    action_list = [-1, 1]
#    gamma = 0.99
#    epsilon = 0.1
#    memory_size = 10000
#    batch_size = 32
#
#    env = gym.make('Pendulm-v0')
#    q_network = Qnetwork(dim_state, action_list, gamma=gamma)
#
#    policy = EpsilonGreedyPolicy(q_network, epsilon=epsilon)
#    memory = []
#    for episode in range(300):
#        state = env.reset()
#        for step in range(200):
#            action, epsilon, q_value = policy.get_action(state, action_list)
#            naxt_state, reward, done, info = env.step([action])
#            if reward < -1:
#                c_reward = -1
#            else:
#                c_reward = 1
#
#        while True:
#            step += 1
#            total_step += 1
#    
#            action = random.choice(action_list)
#            epsilon, q_values = 1.0, None
#    
#            next_state, reward, done, info = env.step([action])
#    
#            if rewrd < -1:
#                c_reward = -1
#            else:
#                c_rewrd = 1
#            memory.append((state, action, c_reward, next_state, done))
#            state = next_state
#    
#            if step > max_step:
#                state = env.reset()
#                step = 0
#            if total_step > n_warmup_steps:
#                break
#        memory = memory[-memory_size:]
#        memory.append((state, action, c_reward, next_state, done))
#        exps = random.sample(memory, batch_size)
#        loss, td_error = q_network.update_on_batch(exps)
#        q_network.sync_target_network(soft=0.01)
#        state = next_state


