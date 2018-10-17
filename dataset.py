#!/usr/bin/env python
# coding=utf-8

import os
import xml.dom.minidom as xml
import tensorflow as tf
from PIL import Image
#root='./train/'
#list=os.listdir(root)
def get_catagory(rootfile):
    catagory=[]
    for i in range(0,len(list)):
        path=rootfile+list[i]
        if os.path.isfile(path) and 'xml' in list[i].split('.'):
            dom=xml.parse(path)
            root=dom.documentElement
            tag=root.getElementsByTagName('name')
            name=tag[0].childNodes[0].data
            if name not in catagory:
                catagory.append(name)
    return catagory

def get_label_location(root,catagory,filename):
    img_path=root+filename
    dom=xml.parse(img_path)
    root=dom.documentElement
    tag_name=root.getElementsByTagName('name')
    for i,name in enumerate(catagory):
        if name==tag_name[0].childNodes[0].data:
            label=i+1
    tag_xmin=root.getElementsByTagName('xmin')
    tag_xmax=root.getElementsByTagName('xmax')
    tag_ymin=root.getElementsByTagName('ymin')
    tag_ymax=root.getElementsByTagName('ymax')
    center_x=float(tag_xmin[0].childNodes[0].data)+(float(tag_xmax[0].childNodes[0].data)-float(tag_xmin[0].childNodes[0].data))/2
    center_y=float(tag_ymin[0].childNodes[0].data)+(float(tag_ymax[0].childNodes[0].data)-float(tag_ymin[0].childNodes[0].data))/2
    w=float(tag_xmax[0].childNodes[0].data)-float(tag_xmin[0].childNodes[0].data)
    h=float(tag_ymax[0].childNodes[0].data)-float(tag_ymin[0].childNodes[0].data)
    return [label,[center_x,center_y,w,h]]


def get_tfrecord(rootfile):
    location=[]
    catagory=get_catagory(rootfile)
    writer=tf.python_io.TFRecordWriter('./train.tfrecords') 
    for img_name in list:
        if 'jpg' in img_name.split('.'):
            img_path=root+img_name
            img=Image.open(img_path)
            img=img.resize((4096,2048))
            img_raw=img.tobytes()
            for split_name in img_name.split('.'):
                if split_name!='jpg':
                    filename=split_name+'.xml'
            label,location=get_label_location(rootfile,catagory,filename)
            print(location)
            #label=ret[0]
            #location.append(ret[1])
            #location.append(ret[2])
            #location.append(ret[3])
            #location.append(ret[4])
            example=tf.train.Example(features=tf.train.Features(feature={
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'location':tf.train.Feature(float_list=tf.train.FloatList(value=location))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def _parse_function(example):
    features={
            'img_raw':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64),
            'location':tf.FixedLenFeature((4),tf.float32)
            }
    parsed_features=tf.parse_single_example(example,features)
    #parsed_features['img_raw']=tf.decode_raw(parsed_features['img_raw'],tf.uint8)
    return parsed_features['img_raw'],parsed_features['label'],parsed_features['location']
                

                        
def get_dataset(filename):
        dataset=tf.data.TFRecordDataset(filename)
        dataset=dataset.map(_parse_function)
        return dataset



if __name__=='__main__':
    
    dataset=get_dataset('./train.tfrecords');
    dataset=dataset.batch(16)
    iterator=dataset.make_one_shot_iterator()
    img,label,location=ext_element=iterator.get_next()
    img=tf.decode_raw(img,tf.uint8)
    with tf.Session() as sess:
        print(sess.run(location))
    
   # get_tfrecord('./train/')


















