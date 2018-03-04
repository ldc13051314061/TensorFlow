#tensorflow 基本运算
#decheng,2018/3/4 21:48

import tensorflow as tf
import numpy as np

#HelloWorld
Word = tf.constant('Hello World')


#加法
#定义op常量
Add1 = tf.constant([[1,2]])
Add2 = tf.constant([[3,4]])
#加法结果
Product_Add = tf.add(Add1,Add2)

#乘法
#占位符,feed方式,使用feed_dict以字典的方式对多个变量输入值。
mul_x = tf.placeholder(tf.int16)
mul_y = tf.placeholder(tf.int16)
mul_xy = tf.multiply(mul_x,mul_y)
#乘法结果
#Product_Mul = tf.matmul(mul_x,mul_y)

#矩阵乘法
a = tf.Variable(tf.ones([3,2]))
b = tf.Variable(tf.ones([2,3]))
product_ab = tf.matmul(a,b)

#自定义W变量
W = tf.Variable(initial_value= np.array([[1,2,3],[4,5,6],[7,8,9]]))

#变量初始化,两种初始化方法都可以
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('My first attempt for tensorflow :' + str(sess.run(Word)))
    print('The add result is' + ' ' + str(sess.run(Product_Add)))
    print('x*y= ',sess.run(mul_xy,feed_dict = {mul_x:5,mul_y:6}))
    print('mat a * b = \n',sess.run(product_ab))
    print(sess.run(W))
