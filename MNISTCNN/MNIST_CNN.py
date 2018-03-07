#使用CNN训练MNIST数据集
#2018.03.06 21:14
#https://www.youtube.com/watch?v=JCBe_yjDmY8&index=27&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def computer_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs, keep_prob :1})#dropout  =1
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict= {xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#normal distrubition
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.constant(0.1,shape=shape)#bias usually>0   =0.1 is good
    return tf.Variable(initial)

def conv2d(x,W):
    #卷积神经网络层，x:图片，2d:二维图片
    #stride[1,x_movement,y_movement,1]
    #Must have strids[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')





def max_pool_2x2(x):
    #X:input：指卷积需要输入的参数，具有这样的shape[batch, in_height, in_width, in_channels]，
    # 分别是[batch张图片, 每张图片高度为in_height, 每张图片宽度为in_width, 图像通道为in_channels]。
    #stride[1,x_movement,y_movement,1]
    #ksize =filter：指用来做卷积的滤波器，当然滤波器也需要有相应参数
    #滤波器的shape为[filter_height, filter_width, in_channels, out_channels]，
    # 分别对应[滤波器高度, 滤波器宽度, 接受图像的通道数, 卷积后通道数]，
    # 其中第三个参数 in_channels需要与input中的第四个参数 in_channels一致，
    # out_channels第一看的话有些不好理解，如rgb输入三通道图，
    # 我们的滤波器的out_channels设为1的话，就是三通道对应值相加，最后输出一个卷积核。
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



#define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])#28*28
ys = tf.placeholder(tf.float32,[None,10])#0-9
keep_prob = tf.placeholder(tf.float32)#dropout

x_image = tf.reshape(xs,[-1,28,28,1])#[样本数-1为不管其为多少，28,28,1通道数，只有黑色为1，grb为3]
print(x_image.shape) #[n_samples,28,28,1]


##conv1 layer
W_conv1 = weight_variable([5,5,1,32])#patch = 5*5 , insize =1 image的厚度为1，outsize 高度  32
b_conv1 = bias_variables([32])
#CNN第一个卷积层
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)) + b_conv1  #output = 28*28*32
#tf.nn.relu 非线性话处理,计算修正线性单元(非常常用)
#ReLU（Rectified Linear unit）激活函数最近变成了神经网络中隐藏层的默认激活函数。
#这个简单的函数包含了返回max(0,x)，所以对于负值，它会返回0，其它返回x。
#pooling
h_pool1 = max_pool_2x2(h_conv1)     #output = 14*14*32

##conv2 layer
W_conv2 = weight_variable([5,5,32,64])  #patch = 5*5,insize = 32,outsize =64
b_conv2 = bias_variables([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)) + b_conv2 #outsize = 14*14*64
h_pool2 = max_pool_2x2(h_conv2)   #outsize = 7*7*64

##fun1 layer
#神经网络层定义，接pooling2后面
W_fc1 = weight_variable([7*7*64,1024])  #1024使其更高
b_fc1 = bias_variables([1024])
#[n_samples,7,7,64] ->> [n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
#有overfitting过拟合处理，加dropout处理
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

##fun2 layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variables([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)


#the erroe between prediction and real data

#交叉信息熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) #loss

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

#important step
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob: 0.5})
    if i % 10 == 0:
        print(computer_accuracy(mnist.test.images,mnist.test.labels))




'''
训练结果
0.1355
0.3128
0.5113
0.6144
0.707
0.7629
0.8045
0.8216
0.8454
0.8354
0.8618
0.8674
0.8826
0.8802
0.8936
0.8962
0.9045
0.8999
0.8965
0.9075
0.9114
0.9182
0.9226
0.9247
0.9276
0.9234
0.9285
0.9191
0.9295
0.9251
0.9286
0.9315
0.9372
0.9327
0.9379
0.9386
0.9385
0.941
0.9326
0.9384
0.9408
0.943
0.9449
0.9445
0.9422
0.9434
0.9426
0.948
0.9474
0.9461
'''


















