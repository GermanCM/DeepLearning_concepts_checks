#%%
import numpy as np 
import pandas as pd 

file_path = r'.\datasets\pima-indians-diabetes.data.csv'
dataset = np.loadtxt(file_path, delimiter=',')

# %%
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# 8 attributes, input_dim=8
model.add(Dense(12, input_dim=8, activation='relu'))
model.add()
# %%
