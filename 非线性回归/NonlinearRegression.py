#非线性回归
#2018.3.6 8:34
#Decheng

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成随机点
x_data = np.linspace(-1,1,200)[:,np.newaxis]
#np.newaxis插入一个维度,本来是生成200个数据为一个行向量，增加一个维度后，变为（200,1），即200行一列的列向量
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise
#y_data = x_data + noise

#定义两个占位符placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间值
#一层神经网络，包括10个神经元
Weight_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weight_L1) + biases_L1
#神经网络输出通过可通过softmax激活函数，其证明参照高斯分布指数族分布函数，吴恩达机器学习教程
L1 = tf.nn.tanh(Wx_plus_b_L1)#双曲正切函数作为激活函数


#定义神经网络输出层
Weight_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))#一个偏置
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2 ) + biases_L2
prediction_y = tf.nn.tanh(Wx_plus_b_L2)


#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction_y))
#使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)
#启动session
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for step in range(20001):#训练次数应该长一些，否则会产生线性效果,为了这个参数调了一上午
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        if step % 20 == 0:
            print('step',step,'loss = ',sess.run(loss,feed_dict={x:x_data,y:y_data}))

    #获得预测值
    prediction_y_value = sess.run(prediction_y,feed_dict={x:x_data})
    print(prediction_y_value)
    #画图
    plt.figure(1)
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_y_value, 'r-',lw = 5)
    plt.xlabel('NolinearRegression')
    plt.show(1)

