# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 09:32:31 2018

@author: Decheng Liu
制作数据集
https://blog.csdn.net/zhangjunp3/article/details/79627824
"""
import os

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
sys.path.append(r'F:\tensorflow Classifier\\')
import TFRecordsRead


# 保存为TFrecords
cwd = r'F:\tensorflow Classifier\Category_Flower\\'
#classes = {'daffodil' , 'snowdrop', 'lilyvalley', 'bluebell'}
#           
#writer = tf.python_io.TFRecordWriter('flower_train.tfrecords') #要生成的文件
#
#for index ,name in enumerate(classes):
#    class_path = cwd + name + '\\'
#    for img_name in os.listdir(class_path):
#        img_path = class_path + img_name  # 每一个图片的地址
#        img = Image.open(img_path)
#        img = img.resize((224,224))
#        plt.imshow(img)
#        print (img_path)
#        img_raw = img.tobytes()   # 将图片转换为二进制
#        example = tf.train.Example(features=tf.train.Features(feature={
#                  'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[index])),
#                  'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#                  }))    #example 对象对label和image数据进行封装
#        writer.write(example.SerializeToString())   # 序列化为字符串
#        plt.show()
#writer.close()


# 显示TFrecord保存的图片和标签
filename_queue = tf.train.string_input_producer(["flower_train.tfrecords"])  # 读入流中
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)  # 返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                             'label':tf.FixedLenFeature([],tf.int64),
                                             'img_raw':tf.FixedLenFeature([],tf.string),
                                            }
                                    )   # 取出包含image和label的feature对象
image = tf.decode_raw(features['img_raw'],tf.uint8)
image = tf.reshape(image,[224,224,3])
label = tf.cast(features['label'],tf.int32)
label = tf.one_hot(label,4,1,0)

with tf.Session() as sess:
    # 开启一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(100):
        example,l = sess.run([image,label])  # 在会话中提取image和label
        img = Image.fromarray(example,'RGB')
        img.save(cwd + str(i) + '_Label_' + str(l) + '.jpg') 
        print(example,l)
    coord.request_stop()
    coord.join(threads)
        















































print ("OKKKKK")





