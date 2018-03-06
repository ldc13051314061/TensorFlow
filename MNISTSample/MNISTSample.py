#2018.3.6 10：48
#MNIST数据集
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#计算每个批次的大小
batch_size = 200
#计算批次数
batch_n = mnist.train.num_examples // batch_size

#建立两个占位符

x = tf.placeholder(tf.float32,[None,784])#28*28图像大小
y = tf.placeholder(tf.float32,[None,10]) #0-9十个标签数

#c创建一个神经网络
W = tf.Variable(tf.zeros([28*28,10]))
b = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.matmul(x,W) + b)

#二次损失函数
loss = tf.reduce_mean(tf.square(y - prediction))

#使用SGD
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化
init = tf.global_variables_initializer()

#结果存放在boolen类型的列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#argmax返回列表中最大数所在的位置索引
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)#初始化全局变量
    for epoch in range(31):
        for batch in range(batch_n):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('epoch' + str(epoch) + 'Test accuracy is ' + str(acc))






