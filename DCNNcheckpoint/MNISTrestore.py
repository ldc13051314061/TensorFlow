#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  :2019-09-10 17:11
# @Author:Decheng liu
'''
重新加载checkpoint

'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 下载数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 测试数据
X_train, Y_train = mnist.train.images, mnist.train.labels
X_test, Y_test = mnist.test.images, mnist.test.labels


# 根据网络结构，定义各个参数，变量并搭建图
tf.reset_default_graph()     # 防止重复定义出错
# 给X，Y定义placeholder, 要去指定的数据类型和形状
inputDim = 784
outputDim = 10
lay1 = 128
lay2 = 64
lay3 = 10
lr = 0.001
std = 0.01
X = tf.placeholder(dtype=tf.float32, shape=[None, inputDim], name="X")  # None 代表样本个数，
Y = tf.placeholder(dtype=tf.float32, shape=[None, outputDim], name="Y")
# 权重初始化
W1 = tf.Variable(tf.random_normal(shape=[inputDim, lay1], mean=0, stddev=std),  name="W1")
b1 = tf.Variable(tf.ones(shape=[lay1]), name="b1")
L1out = tf.nn.relu(tf.matmul(X, W1) + b1, name="L1out")

W2 = tf.Variable(tf.random_normal(shape=[lay1, lay2], mean=0, stddev=std),  name="W2")
b2 = tf.Variable(tf.ones(shape=[lay2]), name="b2")
L2out = tf.nn.relu(tf.matmul(L1out, W2) + b2, name="L2out")

W3 = tf.Variable(tf.random_normal(shape=[lay2, lay3], mean=0, stddev=std),  name="W3")
b3 = tf.Variable(tf.ones(shape=[lay3]), name="b3")
L3out = tf.matmul(L2out, W3) + b3

# cost,logit = log(it) 先进行softmax, 再进行log, 取概率
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=L3out, labels=Y)
loss = tf.reduce_mean(cross_entropy)  # 取均值

# 训练train
trainer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# 重新加载
saver = tf.train.Saver()
save_file = r"./checkpoint1/train_model.ckpt"

# 启动图
with tf.Session() as sess:
    saver.restore(sess, save_path=save_file)
    predictions = tf.equal(tf.argmax(tf.transpose(L3out)), tf.argmax(tf.transpose(Y)))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("training set accuracy: ", sess.run(accuracy, feed_dict={X: X_train, Y: Y_train}))
    print("testing set accuracy: ", sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))
