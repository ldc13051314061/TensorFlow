#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  :2019-09-09 16:25
# @Author:Decheng liu
'''
MNIST
3层神经网络，保存权重文件
我2年前就写过了
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 下载数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 提取数据
# (55000, 784)
# (55000, 10)
# (10000, 784)
# (10000, 10)
X_train, Y_train = mnist.train.images, mnist.train.labels
X_test, Y_test = mnist.test.images, mnist.test.labels
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

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

# 权重文件保存
saver = tf.train.Saver()
save_file = r"./checkpoint1/train_model.ckpt"

# 启动图
with tf.Session() as sess:
    # 所有变量初始化
    sess.run(tf.global_variables_initializer())
    # 定义变量
    Iters = 800
    batch_sample = 64

    # 定义loss列表，便于画图
    loss_item = []
    # 指定迭代次数
    for it in range(Iters):
        # 每次取一批数据集，加快训练
        X_batch, Y_batch = mnist.train.next_batch(batch_size=batch_sample)
        # feed数据，填进数据
        _, loss_batch = sess.run([trainer, loss], feed_dict={X: X_batch, Y: Y_batch})
        loss_item.append(loss_batch)

        # 打印loss
        if it % 100 == 0:
            print("iters: %d, loss: %f" % (it, loss_batch))

    # 保存模型
    saver.save(sess=sess, save_path=save_file)

    print("trained model saved")
    # 训练完成后，分别看训练集和测试机准确率
    predictions = tf.equal(tf.argmax(tf.transpose(L3out)), tf.argmax(tf.transpose(Y)))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("training set accuracy: ", sess.run(accuracy, feed_dict={X: X_train, Y: Y_train}))
    print("testing set accuracy: ", sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))

    # 画出训练准确率
    plt.plot(loss_item, label="train accuracy")
    plt.legend()
    plt.xlabel("iters")
    plt.xticks(np.arange(0, Iters, 100))
    plt.show()


    # 验证，随机抽取照片
    l3out, acc = sess.run([L3out, accuracy], feed_dict={X: X_test, Y: Y_test})
    print("Test set accuracy:", acc)

    # 随机取5张照片
    pic_num = 3
    for i in range(pic_num):
        index = np.random.randint(X_test.shape[0])

        sess.run(tf.Print(l3out[index], ["l3***", l3out[index]], message="输出", summarize=10, name="pprint"))

        predict_num = np.argmax(l3out[index])
        print(L3out[index], predict_num)
        plt.figure(figsize=(5, 5))
        plt.title(predict_num)
        plt.imshow(X_test[index, :].reshape(28, 28))
        plt.show()













