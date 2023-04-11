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

import config

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
        self.board_size = config.board_size
        self.board = np.zeros((config.board_size))  # khởi tạo bàn cờ 3x3 với giá trị ban đầu là 0
        
        self.action_space = spaces.Discrete(config.actions)  # không gian hành động là một số nguyên từ 0 đến 8
        self.observation_space = spaces.Box(low=-1.09973757e+13, high=infinity, shape=(config.board_size, config.board_size), dtype=np.int8)  # không gian quan sát là một ma trận 3x3 với các giá trị từ 0 đến 2
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
            print('----------------- vị trí ({}, {}) ---- quân ----- ({}) trùng----------------'.format(row, col, self.cur_player))
            self.cur_player = self.update_player()
            return self._get_obs(), -1000000, False, {'selected': True}
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
        with open(config.win_loss, mode='w') as f:
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