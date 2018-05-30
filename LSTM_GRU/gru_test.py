import tensorflow as tf;
import numpy as np;

X = tf.random_normal(shape=[1, 5, 6], dtype=tf.float32)
X = tf.reshape(X, [-1, 5, 6])
cell = tf.nn.rnn_cell.GRUCell(8)
init_state = cell.zero_state(1, dtype=tf.float32)
output, state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=False)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    a,b = sess.run([output,state])

    print(a)
    print(b)