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

version_log = str('3_20x20')


def is_empty_cell(board, row, col):
	if board[row][col] == 0:
		return True
	return False

def empty_cells(board):
	emp_cells = []
	for i in range(len(board)):
		for j in range(len(board)):
			if is_empty_cell(board, i, j):
				emp_cells.append([i,j])
	return emp_cells



class CaroEnv(gym.Env):
    def __init__(self):
        self.board_size = 20
        self.board = np.zeros((self.board_size, self.board_size))  # khởi tạo bàn cờ 3x3 với giá trị ban đầu là 0
        
        self.action_space = spaces.Discrete(self.board_size * self.board_size)  # không gian hành động là một số nguyên từ 0 đến 8
        self.observation_space = spaces.Box(low=-1.09973757e+13, high=infinity, shape=(self.board_size, self.board_size), dtype=np.int8)  # không gian quan sát là một ma trận 3x3 với các giá trị từ 0 đến 2
        self.player = 1
        self.opponent = -1
        self.cur_player = self.player
        self.player_win = 0
        self.player_loss = 0
        self.player_draw = 0
    # Lấy trạng thái bàn cờ
    def _get_obs(self):
        return np.copy(self.board)


    # hàm kiểm tra chiến thắng
    def check_winner(self, board, player, oponent):
        rows, cols = board.shape
        
        # Kiểm tra hàng ngang
        for i in range(rows):
            for j in range(cols-4):
                if board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3] == board[i][j+4]:
                    if board[i][j] == player:
                        return 1
                    elif board[i][j] == oponent:
                        return 2
        
        # Kiểm tra hàng dọc
        for i in range(rows-4):
            for j in range(cols):
                if board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j] == board[i+4][j]:
                    if board[i][j] == player:
                        return 1
                    elif board[i][j] == oponent:
                        return 2
        
        # Kiểm tra đường chéo chính
        for i in range(rows-4):
            for j in range(cols-4):
                if board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] == board[i+4][j+4]:
                    if board[i][j] == player:
                        return 1
                    elif board[i][j] == oponent:
                        return 2
        
        # Kiểm tra đường chéo phụ
        for i in range(rows-4):
            for j in range(4, cols):
                if board[i][j] == board[i+1][j-1] == board[i+2][j-2] == board[i+3][j-3] == board[i+4][j-4]:
                    if board[i][j] == player:
                        return 1
                    elif board[i][j] == oponent:
                        return 2
        
        # Kiểm tra hòa
        if 0 not in board:
            return 3
        
        # Trường hợp chưa kết thúc trò chơi
        return 0

    def update_player(self):
        if self.cur_player == self.player:
            return self.opponent
        else:
            return self.player

    def update_player_new_game(self):
        self.player = -self.player
        self.opponent = -self.opponent

    def step(self, row, col):
        if self.board[row][col] != 0:  # kiểm tra nước đi hợp lệ
            print('-----------------vị trí ({}, {}) trùng----------------'.format(row, col))
            return self._get_obs(), -10000000, False, {'selected': True}
        self.board[row][col] = self.cur_player  # cập nhật bàn cờ
        winner = self.check_winner(self.board, self.player, self.opponent)  # kiểm tra xem trò chơi đã kết thúc chưa và xác định người chiến thắng
        
        # Chưa ai win
        if winner == 0:
            reward = evaluate(self.board, self.player, self.opponent)
            done = False
            self.cur_player = self.update_player()
        # Player win
        elif winner == 1: 
            reward = evaluate(self.board, self.player, self.opponent) * 2.0
            print('thắng: ', reward)
            print('player {} win với điểm là {}'.format(self.player, reward))
            self.update_player_new_game()
            self.player_win +=1
            
            done = True
        # Opponent win
        elif winner == 2: 
            reward = evaluate(self.board, self.player, self.opponent) * 2.0
            print('thua: ', reward)
            print('player {} loss với điểm là {}'.format(self.player, reward))
            self.update_player_new_game()
            self.player_loss +=1
            
            done = True
        # Hòa
        elif winner == 3: 
            reward = reward = evaluate(self.board, self.player, self.opponent) * 1.3
            print('hòa: ', reward)
            self.update_player_new_game()
            self.player_draw += 1
            done = True

        info = {'selected': False}
        with open('logs/win_loss_v'+version_log+'.txt', mode='w') as f:
                    f.write("player win : "+ str(self.player_win) + 
                            "\nplayer loss: "+ str(self.player_loss) +
                            "\ndraw: "+ str(self.player_draw))
        return self._get_obs(), reward, done, info
    

    def best_action(self):
        emp = empty_cells(self.board)
        probs = []
        for row, col in emp:
            self.board[row][col] = self.player
            score = evaluate(self.board, self.player, self.opponent)
            self.board[row][col] = 0
            probs.append([row, col, score])
        sort_probs = sorted(probs, key = lambda x: x[2], reverse=True)
        max_prob = sort_probs[0]
        row, col, score = max_prob[0], max_prob[1], max_prob[2]
        return row, col

    # Tạo lại ván cờ mới
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))  # khởi tạo lại bàn cờ
        self.cur_player = self.player  # đặt người chơi
        return np.copy(self.board)

    # Hiển thị bàn cờ
    def render(self, mode='human'):
        board = np.copy(self.board)
        for i in range(len(board)):
            if i == 0:
                print('   ',end=' ')
            print(i, end='  ')
        print('\n')
        w = 0
        for i in board:
            print(w, ' ', end=' ')
            for j in i:
                if(j == 0):
                    print('.', end='  ')
                elif j == 1:
                    print('X', end='  ')
                elif j == -1:
                    print('0', end='  ')
            print(' ')
            w+=1
        print('-------------------------')	


env = CaroEnv()


class DQNAgent:
    def __init__(self, state_size = env.board_size, action_size =  env.board_size *  env.board_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9999999  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.step_update = 0
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # model.add(Conv2D(8, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(3, 3, 1)))
        # model.add(Conv2D(16, kernel_size=(2, 2),padding='same', activation='relu'))
        model.add(Input(shape=(self.state_size,self.state_size,1)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1028, activation='relu'))
        model.add(Dense(2056, activation='relu'))
        model.add(Dense(1028, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            position =  random.randrange(self.action_size)
            return position // self.state_size, position % self.state_size
        state = state.reshape((1, self.state_size, self.state_size, 1))
        act_pred = self.model.predict(state)
        position_pred = np.argmax(act_pred[0])
        row, col = position_pred // self.state_size, position_pred % self.state_size
        print('DQN dự đoán:',row, col)
        return row, col
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        print('len(minibatch)', len(minibatch))
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape((1, self.state_size, self.state_size, 1))
            next_state = next_state.reshape((1,self.state_size, self.state_size, 1))
            position = action[0] * self.state_size + action[1]
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)[0]
            target_f[position] = target
            self.model.fit(state, np.expand_dims(target_f, axis=0), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.gamma *= 0.999
            self.step_update += 1
            with open('logs/num_epsilon_v'+version_log+'.txt', mode='w') as f:
                    f.write("epsilon : "+ str(self.epsilon) + 
                            "\ngamma: "+ str(self.gamma) +
                            "\nstep update model: "+ str(self.step_update))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        # self.model.save(name)


agent = DQNAgent()
try:
    agent.load('model/DQN_v'+version_log+'.h5')
except:
    print('chưa có file')
def train(env, agent, num_episodes=1000, batch_size=64):
    # Khởi tạo list lưu trữ các kết quả trung bình của reward trong 100 episodes gần nhất
    # avg_rewards = []
    
    # Bắt đầu vòng lặp huấn luyện
    for episode in range(num_episodes):
        # Khởi tạo trạng thái đầu tiên
        state = env.reset()
        # print(state)
        # input()
        # Khởi tạo biến lưu trữ tổng reward trong episode
        total_reward = []
        # num = 0
        # Bắt đầu vòng lặp cho mỗi episode
        done = False
        while not done:
            # Chọn hành động dựa trên trạng thái hiện tại
            # print('nguoi choi la hiện tại là: ',env.cur_player)
            # print('nguoi choi là: ',env.player)
            # print('đối thủ là: ',env.opponent)
            # input()
            row, col = agent.act(state)
            
            # Thực hiện hành động và nhận lại thông tin về trạng thái mới, reward và done
            # print('env.cur_player check trùng 1:' , env.cur_player, env.player)
            next_state, reward, done, check_selected = env.step(row, col)
            # print('env.cur_player check trùng 2 :' , env.cur_player, env.player)
            # if check_selected['selected']:
            #     # print('check_selected:', check_selected['selected'])
            #     if env.cur_player == env.player:
            #         agent.remember(state, (row, col), reward, next_state, done)
            #         # print('env.cur_player check trùng 3:' , env.cur_player, env.player)
            #     # input()
            #     # print('reward state', reward)
            #     # env.render(state)
            #     # num+=1
            #         total_reward.append(reward)
            #     row, col = env.best_action()
            #     next_state, reward, done, check_selected = env.step(row, col)
            #     agent.remember(state, (row, col), reward, next_state, done)
            #     # print('env.cur_player check trùng 4:' , env.cur_player, env.player)
            #     # input()
            #     # print('reward next_state', reward)
            #     # print('row col ', row, col)
            #     # env.render(next_state)
            #     # num+=1
            #     total_reward.append(reward)
            #     # input()
            


            # Lưu trữ trạng thái hiện tại, hành động, reward và trạng thái tiếp theo vào bộ nhớ
            agent.remember(state, (row, col), reward, next_state, done)
            
            # Cập nhật trạng thái hiện tại là trạng thái tiếp theo
            state = next_state
            
            # Cộng dồn reward cho tổng reward của episode
            # total_reward += reward
            # num+=1
            total_reward.append(reward)

            # Huấn luyện mô hình nếu số lượng bộ nhớ đủ lớn
            if len(agent.memory) > batch_size + batch_size // 2:
                agent.replay(batch_size)
                agent.save('model/DQN_v'+version_log+'.h5')
                agent.memory.clear()

                
        # Lưu trữ kết quả reward của episode vào list
        # avg_rewards.append(total_reward)

        if episode % 100 == 0:
            print('episode:', episode)
            print('epsilon:', agent.epsilon)

        # print('len(total_reward)', len(total_reward))
        # print('len memory: ', len(agent.memory))
        # print(total_reward)
        with open('logs/history_v1_v'+version_log+'.txt', 'a') as f:
            np.savetxt(f, [np.mean(total_reward)], delimiter=',', footer='', header='')
        with open('logs/so_quan_co_v1_v'+version_log+'.txt', 'a') as f:
            np.savetxt(f, [len(total_reward)], delimiter=',', footer='', header='')



train(env,agent,1000,256)



# nenv = CaroEnv()
# a = nenv.reset()
# nenv.step(5,5)
# nenv.step(10,10)
# nenv.render(a)