import math
import numpy as np
import random
from math import inf as infinity
import gym
from gym import spaces
import numpy as np
from evaluate_20x20 import evaluate
import caro_part1 as caro

import random
from collections import deque
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D
from tensorflow.keras.optimizers import Adam

import config

class DQNAgent:
    def __init__(self, state_size = config.board_size, action_size =  config.board_size *  config.board_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = config.memory
        self.gamma = config.gamma   # discount rate
        self.epsilon = config.epsilon # exploration rate
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate
        self.step_update = config.step_update_model
        self.model = self._build_model()
        self.model_target = self._build_model()
    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size, self.state_size, 1) ))
        # model.add(Conv2D(1, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(self.state_size, self.state_size, 1)))
        # model.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))
        # model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
        # model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
        # model.add(Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(890, activation='relu'))
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        # input()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, player):
        # print('player lafff: ', player)
        if np.random.rand() <= self.epsilon:
            if np.random.rand() <= 0.9999999999999:
                row_1, col_1, _ = caro.set_move_bot(state, player, -player)
                print('vi tri tot:', row_1, col_1)
                return row_1, col_1
            position =  random.randrange(self.action_size)
            return position // self.state_size, position % self.state_size
        state = state.reshape((1, self.state_size, self.state_size, 1))
        act_pred = self.model.predict(state, verbose= 0)
        position_pred = np.argmax(act_pred[0])
        row, col = position_pred // self.state_size, position_pred % self.state_size
        print('DQN dự đoán:',row, col)
        return row, col
    


    def nomalize_reward(self):
        rewards = []
        # print(len(self.memory))
        for _,_,reward,_,_ in self.memory:
            rewards.append(reward)
        # print(rewards)
        # print('len reward: ', len(rewards))
        # input()
        mean = np.mean(rewards, axis=0)
        std = np.std(rewards, axis=0)
        rewards = (rewards - mean) / std
        # min_val = np.min(rewards, axis=0)
        # if min_val < 0 :
        #     min_val = -min_val
        # print(mean)
        # print(std)
        # print(np.sum(rewards, axis=0))
        # print(rewards)
        # print('len rreward: ', len(rewards))
        # print(min_val)
        # rewards = rewards + min_val
        # print(rewards)
        # a = input('check: ')
        # exit()
        for i in range(len(self.memory)):
            new_reward = rewards[i]
            new_tuple = (self.memory[i][0], self.memory[i][1], new_reward, self.memory[i][3], self.memory[i][4])
            self.memory[i] = new_tuple




    def replay(self, batch_size):
        self.nomalize_reward()
        minibatch = random.sample(self.memory, batch_size)
        print('len(minibatch)', len(minibatch))
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape((1, self.state_size, self.state_size, 1))
            next_state = next_state.reshape((1,self.state_size, self.state_size, 1))
            position = action[0] * self.state_size + action[1]
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model_target.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)[0]
            target_f[position] = target
            self.model.fit(state, np.expand_dims(target_f, axis=0), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.gamma *= config.gamma_decay
            self.step_update += 1
            with open(config.num_epsilon, mode='w') as f:
                    f.write("epsilon : "+ str(self.epsilon) + 
                            "\ngamma: "+ str(self.gamma) +
                            "\nstep update model: "+ str(self.step_update))
        if self.step_update % 3 == 0:
            print('target update model')
            self.update_target_model()
    def update_target_model(self):
        self.model_target.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model_target.save_weights(name)
        # self.model.save(name)
