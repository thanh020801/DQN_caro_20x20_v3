import matplotlib.pyplot as plt
import numpy as np
import config 

history = np.loadtxt(config.history_reward)
# print(history)

plt.figure(1)
plt.plot(history, label = "Biểu đồ điểm thưởng")
print(history)
# plt.plot(history.history['val_loss'], label = "validation loss")
plt.title("Điểm thưởng")
plt.xlabel("Số trận")
plt.ylabel("Điểm thưởng")
plt.legend()
plt.show()



so_quan_co = np.loadtxt(config.history_num_chess)
print(so_quan_co)

plt.figure(1)
plt.plot(so_quan_co, label = "biểu đồ số lượng quân cờ")
plt.title("Số quân cờ ")
plt.xlabel("Số trận")
plt.ylabel("Số quân cờ")
plt.legend()
plt.show()


loss = np.loadtxt(config.history_loss)
print(loss)

plt.figure(1)
plt.plot(loss, label = "Biểu đồi lỗi")
plt.title("Lỗi")
plt.xlabel("số lần huấn luyện")
plt.ylabel("chỉ số lỗi")
plt.legend()
plt.show()