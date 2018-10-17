#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from dense_ssd import *
import numpy as np
import dataset  

iteration=100
train_x=tf.placeholder(tf.float32,shape=(None,4096,2048,3))
train_y=tf.placeholder(tf.int64,shape=(None,1))
train_location=tf.placeholder(tf.float32,shape=(None,4))
training_flag=tf.placeholder(tf.bool)

dense_ssd=Densenet_SSD(4,12,training_flag)
anchors=dense_ssd.anchors

gclasses,glocations,gscores=dense_ssd.bboxes_encode(train_y,train_location,anchors)
predictions,locations=dense_ssd.densenet_ssd(train_x)
#predictions=tf.nn.softmax(predictions)
#tf.cast(predictions,tf.int32)
loss=dense_ssd.loss(predictions,locations,gclasses,glocations,gscores)
optimizer=tf.train.AdagradOptimizer(learning_rate=1e-4)
train=optimizer.minimize(loss)
dataset=dataset.get_dataset('./train.tfrecords')
dataset=dataset.shuffle(2)
dataset=dataset.batch(1)
dataset=dataset.repeat(2)
iterator=dataset.make_one_shot_iterator()
initializer=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initializer)
    for i in range(iteration):
        data_x,data_y,data_location=iterator.get_next()
        data_x=tf.decode_raw(data_x,tf.uint8)
        data_x=tf.reshape(data_x,[-1,4096,2048,3])
        data_y=tf.reshape(data_y,[-1,1])
        data_location=tf.reshape(data_location,[-1,4])
        #print(data_x)
        #dic={train_x:data_x,train_y:data_y,train_location:data_location}
        data_x,data_y,data_location=sess.run([data_x,data_y,data_location])
        sess.run(train,feed_dict={train_x:data_x,train_y:data_y,train_location:data_location,training_flag:True})
        print('loss={}'.format(loss))
