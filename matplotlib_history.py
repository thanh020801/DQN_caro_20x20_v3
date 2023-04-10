import matplotlib.pyplot as plt
import numpy as np

history = np.loadtxt('logs/history_v1_v1_20x20.txt')
# print(history)

plt.figure(1)
plt.plot(history, label = "training reward")
print(history)
# plt.plot(history.history['val_loss'], label = "validation loss")
plt.title("Reward Plot")
plt.xlabel("epoch")
plt.ylabel("reward")
plt.legend()
plt.show()



so_quan_co = np.loadtxt('logs/so_quan_co_v1_v1_20x20.txt')
print(so_quan_co)

plt.figure(1)
plt.plot(so_quan_co, label = "training reward")
plt.title("Reward Plot")
plt.xlabel("epoch")
plt.ylabel("reward")
plt.legend()
plt.show()