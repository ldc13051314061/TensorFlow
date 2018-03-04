#tensorflow 线性回归
#decheng,2018/3/4 22:06
#参考https://www.youtube.com/watch?v=-1WcI_Z4iOs&index=6&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#使用numpy生成200个随机点
x_data1 = np.random.rand(200)
y_data = x_data1 * 0.5 + 0.2 #真实值
#加入噪声
noise = np.random.normal(0,0.02,x_data1.shape)
x_data = x_data1 + noise


#构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

#二次代价函数，可以用交叉熵代价函数
loss = tf.reduce_mean(tf.square(y - y_data))
#SGD优化,学习率为0.1
optimizer = tf.train.GradientDescentOptimizer(0.1)
#最小化代价函数
train = optimizer.minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 10 == 0:
            print(step,sess.run([k,b]),'loss =',sess.run(loss))

    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,sess.run(y),'r',lw=5)
    plt.scatter(x_data,noise,s=10, color='y')
    plt.xlabel('x_data')
    plt.ylabel('y')
    plt.text(0.4, 0.1, r'$noise$',fontdict={'size': 16, 'color': 'b'})

    plt.show()

