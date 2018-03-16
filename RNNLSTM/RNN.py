# https://www.zhihu.com/search?type=content&q=RNN%20MNIST
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
#Agg 不行plt.show出不了图
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

print('输入数据')
print(mnist.train.images)#二维数组
print('输入数据shape')
print(mnist.train.images.shape)#(55000,784)即55000个不同的图片，每张图片的像素为28*28=784
print(mnist.train.images[1, :])#显示第一张图片的二维数组

i = 1
im = mnist.train.images[i]
print('[1]', 'im = ', im.shape)  #784列数据

im = im.reshape(-1, 28) #将784个数据整合为28列的二维图片
print('[2]', "im.shape(-1,28)", im.shape)#28*28

plt.imshow(im, cmap='gray')

plt.show()#必须加plt.imshow()
print('图像')

#RNN训练网络
#n_input = 28 ，n_steps= 28
a = np.asanyarray(range(20))
b = a.reshape(-1, 2, 2)
print('生成一列数据\n', a, '\n reshape函数的效果 \n', b)

c = np.transpose(b, [1, 0, 2])  #矩阵维度变化，页 行列 分别为0 1 2
d = c.reshape(-1, 2)
print('\n----------------c-------------------\n')
print(c)
print('\n---------------d---------------------\n')
print(d)


#定义一些模型参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128   #相当于每次训练128张图片
display_step = 100

#networks parameters
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

#tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
#tensorflow LSTM cell require 2xn_hidden length(state & cell)
y = tf.placeholder(tf.float32, [None, n_classes])

#define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))

}

biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True)
_state = lstm_cell.zero_state(batch_size, tf.float32)

#使原始数据的输入与原型匹配
a1 = tf.transpose(x, [1, 0, 2])
a2 = tf.reshape(a1, [-1, n_input])
a3 = tf.matmul(a2,weights['hidden']) + biases['hidden']
a4 = tf.split(a3, n_steps, 0)#对第0个维度进行分割 页：0，行：1，列：2
print('\n')
print('x',  '==========', x.shape, '==============', '\n', x)
print('a1', '==========', a1.shape, '==============', '\n', a1)
print('a2', '==========', a2.shape, '==============', '\n', a2)
print('a3', '==========', a3.shape, '==============', '\n', a3)
print('a4', '==========', a4[1].shape, '==============', '\n', a4)
print('\n')
print('test\n')

# outputs, states = tf.nn.dynamic_rnn(lstm_cell,a4,initial_state=_state,time_major=False)
outputs, states = tf.nn.static_rnn(lstm_cell, a4, initial_state=_state)
print('outputs[-1]')
print(outputs[-1])
print('\n==============================\n')

a5 = tf.matmul(outputs[-1], weights['out']) + biases['out']
print('a5:')
print(a5)
print('\n=================================\n')

#定义cost,使用梯度下降求最优
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=a5))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prd = tf.equal(tf.arg_max(a5, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prd, tf.float32))
init = tf.global_variables_initializer()

#进行模型训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            print("Iter" + str(step * batch_size) + ', Minibatch Loss = ' + '{:.6f}'.format(loss) + ',acc =' + '{:.6f}'.format(acc))

        step += 1

    print("Optimization Finished!")


    #测试模型准确率
    test_len = batch_size
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label = mnist.test.labels[:test_len]

    #评估模型
    correct_pred = tf.equal(tf.arg_max(a5,1),tf.arg_max(y,1))
    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x:test_data,y:test_label}))




'''
Iter12800, Minibatch Loss = 1.880903,acc =0.320312
Iter25600, Minibatch Loss = 1.710962,acc =0.476562
Iter38400, Minibatch Loss = 1.574395,acc =0.515625
Iter51200, Minibatch Loss = 1.328070,acc =0.585938
Iter64000, Minibatch Loss = 1.023922,acc =0.726562
Iter76800, Minibatch Loss = 1.231873,acc =0.625000
Iter89600, Minibatch Loss = 0.955984,acc =0.757812
Optimization Finished!
Testing Accuracy: 0.703125
'''




















