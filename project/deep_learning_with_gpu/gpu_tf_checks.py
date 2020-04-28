#%%
import os
import tensorflow as tf
import cProfile

# %%
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%
tf.executing_eagerly()

# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
import tensorflow
print(tensorflow.__version__)

# %%
