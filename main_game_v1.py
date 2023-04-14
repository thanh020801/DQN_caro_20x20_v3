import numpy as np
import config
from DQN_agent import DQNAgent
from env20x20_v2 import CaroEnv

env = CaroEnv()
agent = DQNAgent()

try:
    agent.load(config.model_name)
    print('###################### Có model #################')
    input('bắt đầu train')
except:
    print('$$$$$$$$$$$$$$$$$$$$$ Chưa có model $$$$$$$$$$$$$$$$$$$$$')

def train(n_episodes, batch_size):
    total_time_step = 0
    for ep in range(n_episodes):
        ep_rewards = []
        state = env.reset()
        done = False
        n_selected = 0
        while not done:
            # env.render(state)
            # input('0')
            total_time_step += 1
            # Cập nhật lại target NN mỗi my_agent.update_targetnn_rate
            if total_time_step % agent.step_update_model == 0:
                # Có thể chọn cách khác = weight của targetnetwork = 0 * weight của targetnetwork  + 1  * weight của mainnetwork
                print('update target model')
                agent.update_target_model()

            if env.cur_player == env.opponent:
                # print('minimax danh:', env.cur_player, env.opponent)
                next_state1, done, action, reward1 = env.step_minimax()
                state = next_state1
                # env.render()
                # input('1')
                if done:
                    # print('done: ', done)
                    print("Ep ", ep+1, " minimax win with reward: ", reward1)
                    env.render()
                    # input('b')
                    # agent.remember(state, action, reward1, next_state1, done)
                    break


            # env.render(state)
            # input('1')
            if env.cur_player == env.player:
                # print('DQN danh:', env.cur_player, env.player)
                row, col = agent.act(state)
                next_state, reward, done, selected = env.step(row, col)
                if selected:
                    # print('selected', n_selected)
                    # input('1')
                    n_selected += 1
                if n_selected == 10:
                    # print('update_player')
                    env.cur_player = env.update_player()
                    n_selected = 0
                    # input('2')
                ep_rewards.append(reward)
                agent.remember(state, (row, col), reward, next_state, done)

                state = next_state
            
                if done:
                    # print('done DQN: ',done)
                    print("Ep ", ep+1, " DQN win with reward: ", np.sum(ep_rewards))
                    env.render()

            # env.render(next_state)
            # input('2')
            
            if len(agent.memory) > batch_size and len(agent.memory) > 0:
                agent.train_main_network(batch_size)
                agent.save(config.model_name)

        if agent.epsilon > agent.epsilon_min and len(agent.memory) > 0:
            agent.epsilon *= agent.epsilon_decay
            agent.gamma *= agent.gamma_decay 



        with open(config.history_reward, 'a') as f:
            np.savetxt(f, [np.mean(ep_rewards)], delimiter=',', footer='', header='')
        with open(config.history_num_chess, 'a') as f:
            np.savetxt(f, [len(ep_rewards)], delimiter=',', footer='', header='')

        with open(config.num_epsilon, mode='w') as f:
                f.write("epsilon = "+ str(agent.epsilon) + 
                        "\ngamma = "+ str(agent.gamma) +
                        "\n\nDQN_win: "+ str(env.player_win) +
                        "\nMinimax_win: "+ str(env.player_loss) +
                        "\nDraw: "+ str(env.player_draw) 
                )

def main():
    train(config.episodes, config.batch_size)
if __name__ == "__main__":
    main()
    

