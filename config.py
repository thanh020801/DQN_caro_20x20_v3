from collections import deque


version_log = str('1')

board_size = 20
actions = board_size * board_size



gamma = 0.95
epsilon = 0.8
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001



step_update_model = 0
memory = deque(maxlen=2000)


episodes = 1000
batch_size = 64



# file
history_reward = 'logs1/history_reward_v'+version_log+'.txt'
history_num_chess = 'logs1/history_num_chess_v'+version_log+'.txt'
win_loss = 'logs1/win_loss_v'+version_log+'.txt'
num_epsilon = 'logs1/num_epsilon_v'+version_log+'.txt'
model_name = 'model1/DQN_v'+version_log+'.h5'