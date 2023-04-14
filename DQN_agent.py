import numpy as np
from math import inf as infinity
import random
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
        self.gamma_decay = config.gamma_decay
        self.epsilon = config.epsilon # exploration rate
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate
        self.step_update_model = config.step_update_model
        self.model = self._build_model()
        self.model_target = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size, self.state_size, 1) ))
        model.add(Flatten())
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(890, activation='relu'))
        # model.add(Dense(2048, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_batch_from_memory(self, batch_size):

        exp_batch = random.sample(self.memory, batch_size)
        state_batch  = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size, self.state_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size, self.state_size)
        done_batch = [batch[4] for batch in exp_batch]
        # print('state_batch', state_batch)
        # print(action_batch)
        # print(reward_batch)
        # print(next_state_batch)
        # print(done_batch)
        # input()
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    



    def act(self, state):
        if np.random.rand() <= self.epsilon:
            position =  random.randrange(self.action_size)
            return position // self.state_size, position % self.state_size
        state = state.reshape((1, self.state_size, self.state_size))
        act_pred = self.model.predict(state, verbose= 0)
        position_pred = np.argmax(act_pred[0])
        row, col = position_pred // self.state_size, position_pred % self.state_size
        print('DQN dự đoán:',row, col)
        return row, col
    


    def nomalize_reward(self):
        rewards = []
        for _,_,reward,_,_ in self.memory:
            rewards.append(reward)

        mean = np.mean(rewards, axis=0)
        std = np.std(rewards, axis=0)
        rewards = (rewards - mean) / std

        for i in range(len(self.memory)):
            new_reward = rewards[i]
            new_tuple = (self.memory[i][0], self.memory[i][1], new_reward, self.memory[i][3], self.memory[i][4])
            self.memory[i] = new_tuple

    def train_main_network(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.get_batch_from_memory(batch_size)

        # Lấy Q value của state hiện tại
        q_values = self.model.predict(state_batch, verbose=0)
        
        # Lấy Max Q values của state S' (State chuyển từ S với action A)
        next_q_values = self.model_target.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q_values, axis=1)
        
        for i in range(batch_size):
            new_q_values = reward_batch[i] if done_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]
            position = action_batch[i][0] * self.state_size + action_batch[i][1]

            q_values[i][position] = new_q_values
            

        history = self.model.fit(state_batch, q_values, verbose=0)
        # print(history.history['loss'])
        # input()
        with open(config.history_loss, 'a') as f:
            np.savetxt(f, history.history['loss'], delimiter=',', footer='', header='')

    # def replay(self, batch_size):
    #     # self.nomalize_reward()
    #     minibatch = random.sample(self.memory, batch_size)
    #     print('len(minibatch)', len(minibatch))
    #     for state, action, reward, next_state, done in minibatch:
    #         state = state.reshape((1, self.state_size, self.state_size, 1))
    #         next_state = next_state.reshape((1,self.state_size, self.state_size, 1))
    #         position = action[0] * self.state_size + action[1]
    #         target = reward
    #         if not done:
    #             target = (reward + self.gamma *
    #                       np.amax(self.model_target.predict(next_state, verbose=0)[0]))
    #         target_f = self.model.predict(state, verbose=0)[0]
    #         target_f[position] = target
    #         self.model.fit(state, np.expand_dims(target_f, axis=0), epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    #         self.gamma *= config.gamma_decay
    #         self.step_update += 1
    #         with open(config.num_epsilon, mode='w') as f:
    #                 f.write("epsilon : "+ str(self.epsilon) + 
    #                         "\ngamma: "+ str(self.gamma) +
    #                         "\nstep update model: "+ str(self.step_update))
    #     if self.step_update % 3 == 0:
    #         print('target update model')
    #         self.update_target_model()




    def update_target_model(self):
        self.model_target.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)
        self.model_target.load_weights(name)

    def save(self, name):
        self.model_target.save_weights(name)
        # self.model.save(name)
