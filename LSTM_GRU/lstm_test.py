import tensorflow as tf;
import numpy as np;
from tensorflow.contrib import rnn
X = tf.random_normal(shape=[1, 4, 5], dtype=tf.float32)
X = tf.reshape(X, [-1, 4, 5])
cell = rnn.BasicLSTMCell(6)
init_state = cell.zero_state(1, dtype=tf.float32)
outputs,final_state = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    outputs, final_state = sess.run([outputs,final_state])
    # final_state[0]是cell state  为h
    # final_state[1]是hidden_state  与最后一个output相等  即 output[-1] = final_state[1]
    print(outputs)
    print(final_state)