from keras.utils import plot_model
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections.abc import Iterable
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D
from tensorflow.keras.optimizers import Adam

from DQN_agent import DQNAgent
import config
agent = DQNAgent()
agent.load(config.model_name)
# Plot the model
plot_model(agent.model, to_file='model_plot_v3.png', show_shapes=True, show_layer_names=True)

# Show the plot
img = plt.imread('model_plot_v3.png')
plt.imshow(img)
plt.axis('off')
plt.show()



# from keras.models import load_model
# from vis.visualization import visualize_activation

# # Load model
# model = _build_model()
# # model.load_weigths('model/DQN_v3_20x20.h5')
# # model = load_model('model/DQN_v3_20x20.h5')

# # Visualize activations of layer 2
# layer_idx = 2
# filter_idx = None
# img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
