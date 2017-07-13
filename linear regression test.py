from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
rng = np.random

# Parameters
learning_rate = 0.001
training_epochs = 200
display_step = 10

a_train=np.zeros(shape=(1,1))
b_train=np.zeros(shape=(1,1))
with open('self_send') as f:
    for line in islice(f, 25000, 25100):
        com1=complex(line[2:-2].replace("+-", "-"))
        a_train = np.append(a_train, [com1])
print(a_train)
with open('self_receive') as f:
    for line in islice(f, 25000, 25100):
        com2 = complex(line[2:-2].replace("+-", "-"))
        b_train = np.append(b_train, [com2])
print(b_train)
a_train_re=a_train.real
a_train_im=a_train.imag

b_train_re=b_train.real
b_train_im=b_train.imag


n_samples = a_train.shape[0]


# tf Graph Input
X_rel = tf.placeholder(tf.float32)
X_im  = tf.placeholder(tf.float32)
Y_rel = tf.placeholder(tf.float32)
Y_im  = tf.placeholder(tf.float32)

Saver=tf.train.import_meta_graph('linear_regression_test-1000.meta')
sess1=tf.Session()
Saver.restore(sess1,tf.train.latest_checkpoint('./'))


# Set model weights
W = tf.Variable(sess1.run('weight:0'), name="weight")
print("W=",sess1.run('weight:0'))
b_rel = tf.Variable(0.0, name="bias")
b_im  = tf.Variable(0.0, name="biasIm")

# Construct a linear model
pred_rel = tf.add(tf.multiply(X_rel, W), b_rel)
pred_im  = tf.add(tf.multiply(X_im, W), b_im)

cost = tf.reduce_sum((tf.pow(pred_rel-Y_rel, 2))+(tf.pow(pred_im-Y_im,2)),name="loss")

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
Saver1=tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x_real, x_im, y_real, y_im) in zip(a_train_re, a_train_im, b_train_re, b_train_im):
            sess.run(optimizer, feed_dict={X_rel: x_real, X_im: x_im, Y_rel: y_real, Y_im: y_im})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X_rel: x_real, X_im: x_im, Y_rel: y_real, Y_im: y_im})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b_rel=", sess.run(b_rel),"b_im=", sess.run(b_im))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X_rel: x_real, X_im: x_im, Y_rel: y_real, Y_im: y_im})
    W_result=sess.run(W)
    print("Training cost=", training_cost, "W=", sess.run(W), "b_rel=", sess.run(b_rel),"b_im=", sess.run(b_im), '\n')
    Saver1.save(sess, 'linear_regression_test', global_step=1000)
    print("Model Saved!")
    #Saver.restore(sess, tf.train.latest_checkpoint('./'))
    # Graphic display
    plt.plot(a_train_re, b_train_re, 'ro', label='Original data Real Part')
    plt.plot(a_train_im, b_train_im, 'bs', label='Original data Im Part')
    plt.plot(a_train_re, sess.run(W) * a_train_re + sess.run(b_rel), label='Fitted line_Re')
    plt.plot(a_train_im, sess.run(W) * a_train_im + sess.run(b_im), label='Fitted line_Im')
    plt.legend()
    plt.show()

print("W=",W_result)