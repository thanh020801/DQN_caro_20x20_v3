from collections import deque


version_log = str('2')

board_size = 20
actions = board_size * board_size
win_state = 5

epsilon = 0.9999999999
gamma = 0.95
step_update_model = 25

# epsilon : 0.838940600140004
# gamma: 0.838109230801361
# step update model: 25


gamma_decay = 0.995
epsilon_min = 0.01
epsilon_decay = 0.993
learning_rate = 0.001


# epsilon : 0.7700431458051551
# gamma: 0.8339186846473542
# step update model: 26


# epsilon : 0.5811664141181095
# gamma: 0.8256003457679969
# step update model: 54

# epsilon : 0.4342313267918117
# gamma: 0.8214723440391569
# step update model: 83

# epsilon : 0.31480917318095203
# gamma: 0.6997316702628097
# step update model: 115

# epsilon : 0.28186069554046345
# gamma: 0.7774041149715403
# step update model: 126


memory = deque(maxlen=2000)


episodes = 1000
batch_size = 128



# file
history_reward = 'logs1/history_reward_v'+version_log+'.txt'
history_num_chess = 'logs1/history_num_chess_v'+version_log+'.txt'
win_loss = 'logs1/win_loss_v'+version_log+'.txt'
num_epsilon = 'logs1/num_epsilon_v'+version_log+'.txt'
model_name = 'model1/DQN_v'+version_log+'.h5'