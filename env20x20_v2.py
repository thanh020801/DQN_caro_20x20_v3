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

import caro_part1 as caro
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
        reward = 0
        done = False
        # lượt của đối thủ
        print('cur _ player là: ', self.cur_player)
        print('sss player là: ', self.player)
        input('1')
        if self.cur_player == self.opponent:
            # tìm ô trống và đánh random
            row_1, col_1, _ = caro.set_move_bot(self._get_obs(), self.cur_player, -self.cur_player)
            # print('row, col', row_1, col_1)
            # able_cells = empty_cells(self._get_obs())
            # row_1, col_1 = random.sample(able_cells, 1)[0]
            self.board[row_1][col_1] = self.cur_player
            
            # nếu win hay hòa thì return về bàn cờ và điểm số, ngược lại nếu thì cập nhật cur_player 
            winner = self.check_winner(self._get_obs(), self.player, self.opponent)
            if winner == 2 or winner == 3:
                reward1 = abs(evaluate(self._get_obs(), self.player, self.opponent)) * -2.0
                # print('thua: ', reward1)
                print('player {} thua với điểm là {}'.format(self.player, reward1))
                self.player_loss +=1
                return self._get_obs(), reward1, True, {}
            input('2')
            self.cur_player = self.update_player()

        # lượt của người chơi
        if self.cur_player == self.player:
            # kiểm tra quân cờ hợp lệ nếu ko hợp lệ thì return
            input('3')
            if not is_empty_cell(self._get_obs(), row, col):
                print('----------------- vị trí ({}, {}) ---- quân ----- ({}) trùng----------------'.format(row, col, self.cur_player))
                reward = evaluate(self.board, self.player, self.opponent) - 1000000
                self.cur_player = self.update_player()
                return self._get_obs(), reward, False, {'selected': True}
            input('33')
            # cập nhật quân cờ. kiểm tra thắng thua hòa và tính điểm cho trạng thái
            self.board[row][col] = self.cur_player
            winner = self.check_winner(self._get_obs(), self.player, self.opponent)
            if winner == 0:
                reward = evaluate(self.board, self.player, self.opponent)
                done = False
                self.cur_player = self.update_player()
            elif winner == 1:
                reward = evaluate(self.board, self.player, self.opponent) * 2.0
                # print('thắng: ', reward)
                print('player {} win với điểm là {}'.format(self.player, reward))
                self.player_win +=1
                done = True

            elif winner == 3: 
                reward = evaluate(self.board, self.player, self.opponent) * 1.3
                print('hòa: ', reward)
                self.player_draw += 1
                done = True

        
            with open(config.win_loss, mode='w') as f:
                f.write("player win : "+ str(self.player_win) + 
                        "\nplayer loss: "+ str(self.player_loss) +
                        "\ndraw: "+ str(self.player_draw))
            
            return self._get_obs(), reward, done, {}
        input('4')
        print('chuuuuuuuuuuu')



    

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
        self.update_player_new_game()
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