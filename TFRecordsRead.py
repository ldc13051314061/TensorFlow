# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:27:26 2018

@author: Decheng Liu
读取文件
"""

# 读取TFrecord文件
import tensorflow as tf

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)    # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                                 'label':tf.FixedLenFeature([],tf.int64),
                                                 'img_raw':tf.FixedLenFeature([], tf.string),
                                                })   # 取出image数据和label数据
    img = tf.decode_raw(features['imh_raw'],tf.uint8)
    img = tf.reshape(img,[224,224,3])
    img = tf.cast(img,tf.float32) * (1./225) -0.5  # 在流中抛出img张量
    label = tf.cast(features['label'],tf.int32) # 在流中抛出label张量
    return img,label
    
    

print('OKKKKKK')    
    