import pygame

import numpy as np
from tensorflow import keras

import config
from env20x20_v2 import CaroEnv, empty_cells, is_empty_cell
from DQN_agent import DQNAgent
agent = DQNAgent()
env = CaroEnv()

board =  env.reset()

try:
    agent.load(config.model_name)
    print('###################### Có model #################')
except:
    print('$$$$$$$$$$$$$$$$$$$$$ Chưa có model $$$$$$$$$$$$$$$$$$$$$')


def probability_positions(board, prediction):
    emp = empty_cells(board)
    probs = []
    for row, col in emp:
        position = row * env.board_size + col
        prob_position = prediction[position] 
        probs.append([row, col, prob_position])
    sort_probs = sorted(probs, key = lambda x: x[2], reverse=True)
    max_prob = sort_probs[0]
    return max_prob[0], max_prob[1]


pygame.init()


# Thiết lập màn hình
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Game caro 20x20")


# Màu sắc
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

GREEN = (0, 255, 0)
RED = ( 195, 238, 155)
A = (44, 45, 46)
screen.fill(A)
# Vẽ lưới caro
block_size = 200  # kích thước mỗi ô vuông
for x in range(0, 600, block_size):
    for y in range(0, 600, block_size):
        rect = pygame.Rect(x, y, block_size, block_size)
        pygame.draw.rect(screen, RED, rect, 1)


# Vẽ quân cờ
def draw_x(x, y):
    pygame.draw.line(screen, WHITE, (x, y), (x + block_size-3, y + block_size-3), 3)
    pygame.draw.line(screen, WHITE, (x + block_size, y), (x, y + block_size), 3)

def draw_o(x, y):
    pygame.draw.circle(screen, GREEN, (x + block_size//2, y + block_size//2), block_size//2 - 3, 3)
    
# Khởi tạo ma trận 20x20 để lưu vị trí các quân cờ

# Vẽ quân cờ
def draw_board(board):
    for i in range(env.board_size):
        for j in range(env.board_size):
            if board[i][j] == 1:
                draw_x(j * block_size, i * block_size)
            elif board[i][j] == -1:
                draw_o(j * block_size, i * block_size)

human = env.opponent
dqn = env.player
env.cur_player = human

# Xử lý sự kiện
def handle_events(player):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        elif player == dqn:
            cur_board = np.reshape(env._get_obs(), [1,env.board_size,env.board_size,1])
            pred = agent.model_target.predict(cur_board, verbose = 0)[0]
            pred = np.argmax(pred,axis = 0)
            row, col = divmod(pred,env.board_size)
            print("row: ", row, ",col: ", col)
            # row, col = probability_positions(env._get_obs(), pred)
            if env.board[row][col] == 0:
                env.board[row][col] = player
            print(row, col)
            draw_board(env._get_obs())
            env.cur_player = env.update_player()

        elif player == human:
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                row, col = y // block_size, x // block_size
                if is_empty_cell(env._get_obs(), row, col):
                    env.board[row][col] = player
                    draw_board(board)
                    env.cur_player = env.update_player()
            else:
                continue
        else:
                continue
        




# BOT.chess = 1
# HUMAN.chess = -1
def game():
    while True:
        # show_chess_board(board)
        handle_events(env.cur_player)
        pygame.display.update()
        winner = env.check_winner(env._get_obs(), dqn, human)
        if winner != 0:
            if winner == 1: 
                print('DQN win')
                env.render()
            if winner == 2:
                print('Người chơi win')
                env.render()
            if winner == 3: 
                print('Hòa')
                env.render()
            print('Ket thuc van')
            break

game()

