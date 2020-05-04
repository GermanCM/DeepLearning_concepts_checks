#%%
import tensorflow as tf 
import numpy as np 
from numpy import random
import time

def prepare_mnist_features_and_labels(x, y):
    import tensorflow as tf
    
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

def mnist_dataset():
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist

    # load dataset 
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    print('LONGITUD TRAIN SET: ', len(x))
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds

#%%
train_dataset = mnist_dataset()

import tensorflow

model = tensorflow.keras.Sequential((
    tensorflow.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tensorflow.keras.layers.Dense(100, activation='relu'),
    tensorflow.keras.layers.Dense(100, activation='relu'),
    tensorflow.keras.layers.Dense(10)))

#%%
model.build()
optimizer = tensorflow.keras.optimizers.Adam()

compute_loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy()
#%%
def train_one_step(model, optimizer, x, y):
    import tensorflow as tf

    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    compute_accuracy(y, logits)
    return loss

'''
The tf.function decorator
When you annotate a function with tf.function, you can still call it like any other function. 
But it will be compiled into a graph, which means you get the benefits of faster execution, running on GPU or TPU, or exporting to SavedModel.
'''
@tf.function
def train(model, optimizer):
    train_ds = mnist_dataset()
    step = 0
    loss = 0.0
    accuracy = 0.0
    for x, y in train_ds:
        step += 1
        loss = train_one_step(model, optimizer, x, y)
        if step % 10 == 0:
            tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
    return step, loss, accuracy

init_time = time.time()
step, loss, accuracy = train(model, optimizer)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())

# %%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%

# %%
