# coding: utf-8

# # Logistic Regression with TensorFlow
# By Ivan

# ## 0. Imports

# In[ ]:

import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# ## 1. Download MNIST dataset

# In[ ]:

# Using Tensorflow's default tools to fetch data, this is the same as what we did in the first homework assignment.
mnist = input_data.read_data_sets('./mnist', one_hot=True)

# In[ ]:

# Check the dimension of training, validation and test sets.
mnist.train.images.shape

# In[ ]:

mnist.train.labels.shape

# In[ ]:

mnist.validation.images.shape

# In[ ]:

mnist.validation.labels.shape

# In[ ]:

mnist.test.images.shape

# In[ ]:

mnist.test.labels.shape

# ## 2. Model initialization

# In[ ]:

# Random seed.
rseed = 42
batch_size = 200
lr = 1e-1
num_epochs = 50
num_train, num_feats = mnist.train.images.shape
num_test = mnist.test.images.shape[0]
num_classes = mnist.train.labels.shape[1]
num_hider = 500

# In[ ]:

# Placeholders that should be filled with training pairs (x, y). Use None to unspecify the first dimension
# for flexibility.
train_x = tf.placeholder(tf.float32, [None, num_feats], name="train_x")
train_y = tf.placeholder(tf.int32, [None, num_classes], name="train_y")
# Model weights of logistic regression.
w1 = tf.Variable(tf.random_normal(shape=[num_feats, num_hider], stddev=0.1), name="lr_weights_1")
b1 = tf.Variable(tf.zeros([num_hider]), name="lr_bias_1")
w2 = tf.Variable(tf.random_normal(shape=[num_hider, num_classes], stddev=0.1), name="lr_weights_2")
b2 = tf.Variable(tf.zeros([num_classes]), name="lr_bias_2")

# ## 3. Model training and testing

# In[ ]:

# logits is the log-probablity of each classes.
l1 = tf.nn.relu(tf.matmul(train_x, w1) + b1)
l2 = tf.matmul(l1, w2) + b2
# Use TensorFlow's default implementation to compute the cross-entropy loss of classification.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=l2, labels=train_y, name="loss")
loss = tf.reduce_mean(cross_entropy)
# Build prediction function.
preds = tf.nn.softmax(l2)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(train_y, 1))
# Need to cast the type of correct_preds to float32 in order to compute the average mean accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# In[ ]:

# Use TensorFlow's default implementation for optimziation algorithm. Note that we can understand
# an optimization procedure as an OP (operator) as well.
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# In[ ]:

# Start training!
num_batches = num_train / batch_size
losses = []
train_accs, valid_accs = [], []
time_start = time.time()
with tf.Session() as sess:
    # Visualize the process in TensorBoard.
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    # Before evaluating the graph, we should initialize all the variables.
    sess.run(tf.global_variables_initializer())
    for i in range(num_epochs):
        # Each training epoch contains num_batches of parameter updates.
        total_loss = 0.0
        for _ in range(int(num_batches)):
            # Fetch next mini-batch of data using TensorFlow's default method.
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            # Note that we also need to include optimizer into the list in order to update parameters, but we
            # don't need the return value of optimizer.
            _, loss_batch = sess.run([optimizer, loss], feed_dict={train_x: x_batch, train_y: y_batch})
            total_loss += loss_batch
        # Compute training set and validation set accuracy after each epoch.
        train_acc = sess.run([accuracy], feed_dict={train_x: mnist.train.images, train_y: mnist.train.labels})
        valid_acc = sess.run([accuracy], feed_dict={train_x: mnist.validation.images, train_y: mnist.validation.labels})
        losses.append(total_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        print
        "Number of iteration: {}, total_loss = {}, train accuracy = {}, validation accuracy = {}".format(i, total_loss,
                                                                                                         train_acc,
                                                                                                         valid_acc)
    # Evaluate the test set accuracy at the end.
    test_acc = sess.run([accuracy], feed_dict={train_x: mnist.test.images, train_y: mnist.test.labels})
time_end = time.time()
print("Time used for training = {} seconds.".format(time_end - time_start))
print("MNIST image classification accuracy on test set = {}".format(test_acc))

# In[ ]:

# Plot the losses during training.
plt.figure()
plt.title("Logistic regression with TensorFlow")
plt.plot(losses, "b-o", linewidth=2)
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Cross-entropy")
plt.show()

