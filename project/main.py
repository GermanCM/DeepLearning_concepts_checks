# Main script where to test simple Tensorflow and Keras examples
#%%
'''
import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
sess = tf.Session() --> https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md 
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
'''
