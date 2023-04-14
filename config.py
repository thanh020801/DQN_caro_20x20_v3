from collections import deque


version_log = str('3x3_v5')

board_size = 3
actions = board_size * board_size
win_state = 3

epsilon = 0.5
gamma = 0.8
step_update_model = 30

gamma_decay = 0.9999
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001


memory = deque(maxlen=10000)

episodes = 500
batch_size = 128


history_reward = 'logs2/history_reward_v'+version_log+'.txt'
history_loss = 'logs2/history_loss_v'+version_log+'.txt'
history_num_chess = 'logs2/history_num_chess_v'+version_log+'.txt'
win_loss = 'logs2/win_loss_v'+version_log+'.txt'
num_epsilon = 'logs2/num_epsilon_v'+version_log+'.txt'
model_name = 'model2/DQN_v'+version_log+'.h5'