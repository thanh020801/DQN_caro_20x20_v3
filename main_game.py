import math
import numpy as np
import random
from math import inf as infinity
import gym
from gym import spaces
import numpy as np
from evaluate_20x20 import evaluate

import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D
from tensorflow.keras.optimizers import Adam

import config
from DQN_agent import DQNAgent
from env20x20_v2 import CaroEnv

env = CaroEnv()
agent = DQNAgent()

try:
    agent.load(config.model_name)
except:
    print('chưa có file')
def train(env, agent, num_episodes, batch_size):
    # Khởi tạo list lưu trữ các kết quả trung bình của reward trong 100 episodes gần nhất
    # avg_rewards = []
    
    # Bắt đầu vòng lặp huấn luyện
    for episode in range(num_episodes):
        # Khởi tạo trạng thái đầu tiên
        state = env.reset()
        # print('1cur _ player là: ', env.cur_player)
        # print('1sss player là: ', env.player)
        
        # input()
        # Khởi tạo biến lưu trữ tổng reward trong episode
        total_reward = []
        # num = 0
        # Bắt đầu vòng lặp cho mỗi episode
        done = False
        while not done:
            # Chọn hành động dựa trên trạng thái hiện tại
            
            # elif env.cur_player == env.opponent:
            if env.cur_player == env.opponent:
                next_state, reward, done, _ = env.step_minimax()
                state = next_state
            if env.cur_player == env.player:
                # print('env player', env.player)
                row, col = agent.act(state, env.player)
                next_state, reward, done, _ = env.step(row, col)
            # Thực hiện hành động và nhận lại thông tin về trạng thái mới, reward và done
            
            # env.render()
            # input()
            # Lưu trữ trạng thái hiện tại, hành động, reward và trạng thái tiếp theo vào bộ nhớ
            agent.remember(state, (row, col), reward, next_state, done)
            
            # Cập nhật trạng thái hiện tại là trạng thái tiếp theo
            state = next_state
            
            # Cộng dồn reward cho tổng reward của episode
            # total_reward += reward
            # num+=1
            total_reward.append(reward)

            # Huấn luyện mô hình nếu số lượng bộ nhớ đủ lớn
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)
                agent.save(config.model_name)
                # agent.memory.clear()

                
        # Lưu trữ kết quả reward của episode vào list
        # avg_rewards.append(total_reward)

        if episode % 100 == 0:
            print('episode:', episode)
            print('epsilon:', agent.epsilon)

        # print('len(total_reward)', len(total_reward))
        # print('len memory: ', len(agent.memory))
        # print(total_reward)
        # print('np.mean(total_reward):',np.mean(total_reward))
        # print('len reward: ', len(total_reward))
        with open(config.history_reward, 'a') as f:
            np.savetxt(f, [np.mean(total_reward)], delimiter=',', footer='', header='')
        with open(config.history_num_chess, 'a') as f:
            np.savetxt(f, [len(total_reward)], delimiter=',', footer='', header='')



def main():
    train(env, agent, config.episodes, config.batch_size)
if __name__ == "__main__":
    main()
    

