import matplotlib.pyplot as plt
import numpy as np
import config 

history = np.loadtxt(config.history_reward)
# print(history)

plt.figure(1)
plt.plot(history, label = "training reward")
print(history)
# plt.plot(history.history['val_loss'], label = "validation loss")
plt.title("Reward Plot")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.show()



so_quan_co = np.loadtxt(config.history_num_chess)
print(so_quan_co)

plt.figure(1)
plt.plot(so_quan_co, label = "training số lượng quân cờ")
plt.title("Số quân cờ ")
plt.xlabel("episode")
plt.ylabel("số quân cờ")
plt.legend()
plt.show()