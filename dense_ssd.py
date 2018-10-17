#!/usr/bin/env python
# coding=utf-8
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm,flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
import custom_layers
import ssd_common
import tf_extended as tfe
#Hyperparameter
growth_k=24
nb_blocks=2
init_learning_rate=1e-4
epsilon=1e-4
dropout_rate=0.2

#using Momentum Optiizer
nesterov_momentum=0.9
weight_decay=1e-4

#Label and batch_size
batch_size=16
iteration=200 #batch_size*iteration=dataset_num

test_iteration=10
total_epochs=300

def conv_layer(input_x,_filter,kernel,stride=1,layer_name='conv'):
    with tf.name_scope(layer_name):
        network=tf.layers.conv2d(inputs=input_x,filters=_filter,kernel_size=kernel,strides=stride,padding='SAME',use_bias=False)
        return network

def Global_Average_Pooling(x,stride=1):
    width=np.shape(x)[1]
    height=np.shape(x)[2]
    pool_size=[width,height]
    return tf.layers.average_pooling2d(inputs=x,pool_size=pool_size,strides=stride)

def Batch_Normalization(x,training,scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda:batch_norm(inputs=x,is_training=training,reuse=None),
                       lambda:batch_norm(inputs=x,is_training=training,reuse=True))


def Drop_out(x,rate,training):
    return tf.layers.dropout(inputs=x,rate=rate,training=training)

def Relu(x):
    return tf.nn.relu(x)

# origin is [2,2] stride=2
def Average_pooling(x,pool_size=[4,4],stride=4,padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

def Max_pooling(x,pool_size=[3,3],stride=2,padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

def Concatenation(layers):
    return tf.concat(layers,axis=3)

def Linear(x):
    return tf.layers.dense(inputs=x,units=class_num,name='linear')

"""
def Evaluate(sess):
    test_acc=0.0
    test_loss=0.0
    test_pre_index=0
    add=30  #every step add

    for it in range(test_iteration):
        test_batch_x=test_x[test_pre_index:test_pre_index+add]
        test_batch_y=test_y[test_pre_index:test_pre_index+add]
        test_pre_index+=add
        
        test_feed_dict={
            x:test_batch_x,
            label:test_batch_y,
            learning_rate:epoch_learning_rate,
            training_flag:False
        }
        loss_,acc_=sess.run([cost,accuracy],feed_dict=test_feed_dict)

        test_loss+=loss_/10.0
        test_acc+=acc_/10.0
    summary=tf.Summary(value=[tf.Summary.Value(tag='test_loss',
                                               simple_value=test_loss),
                              tf.Summary.Value(tag='test_accuracy',
                                               simple_value=test_acc)])
    return test_acc,test_loss,sumamry

"""

class Densenet_SSD():
    def __init__(self,nb_blocks,filters,training):
        self.nb_blocks=nb_blocks
        self.filters=filters
        self.training=training
        self.dtype=tf.float32
        self.num_classes=60
        self.image_shape=[4096,2048]
        self.feat_layers=['block1','block2','block3','block4']
        self.feat_shapes=[(512,256),(128,64),(32,16),(8,4)]
        self.anchor_sizes=[(273.,307.,),(307.,675.),(675.,1044.),(1044.,1413.)]
        self.anchor_ratios=[[2,.5],[2,.5,3,1./3],[2,.5],[2,.5]]
        self.anchor_steps=[8,32,128,512]
        self.anchor_offset=0.5
        self.anchors=self.anchors(self.image_shape)



    def bottleneck_layer(self,x,scope):
        with tf.name_scope(scope):
            x=Batch_Normalization(x,training=self.training,scope=scope+'_batch1')
            x=Relu(x)
            x=conv_layer(x,_filter=4*self.filters,kernel=[1,1],layer_name=scope+'_conv1')
            x=Drop_out(x,rate=dropout_rate,training=self.training)
            x=Batch_Normalization(x,training=self.training,scope=scope+'_batch2')
            x=Relu(x)
            x=conv_layer(x,_filter=self.filters,kernel=[6,6],layer_name=scope+'_conv2') 
            # because so big picture ,the kernel should be bigger than 3? set to 6
            x=Drop_out(x,rate=dropout_rate,training=self.training)

            return x
    def transition_layer(self,x,scope):
        with tf.name_scope(scope):
            x=Batch_Normalization(x,training=self.training,scope=scope+'_batch1')
            x=Relu(x)
            x=conv_layer(x,_filter=self.filters,kernel=[1,1],layer_name=scope+'_conv1')
            x=Drop_out(x,rate=dropout_rate,training=self.training) 
            # because the perception filed is bigger ,so stride is 4
            x=Average_pooling(x,pool_size=[2,2],stride=2)

            return x

    def dense_block(self,input_x,nb_layers,layer_name):
        with tf.name_scope(layer_name):
            layers_concat=list()
            layers_concat.append(input_x)
            x=self.bottleneck_layer(input_x,scope=layer_name+'_bottleN_'+str(0))
            layers_concat.append(x)

            for i in range(nb_layers-1):
                x=Concatenation(layers_concat)
                x=self.bottleneck_layer(x,scope=layer_name+'_bottleN_'+str(i+1))
                layers_concat.append(x)
            x=Concatenation(layers_concat)
            return x
    
    def tensor_shape(self,x,rank=3):
        """""
        return the dimensions of a tensor
        """""
        if x.get_shape().is_fully_defined():
            return x.get_shape().as_list()
        else:
            static_shape=x.get_shape().with_rank(rank).as_list()
            dynamic_shape=tf.unstack(tf.shape(x),rank)
            return [s if s is not None else d for s,d in zip(static_shape,dynamic_shape)]


    def anchors(self,img_shape,dtype=np.float32):
        return self.ssd_anchors_all_layers(
                    img_shape,
                    self.feat_shapes,
                    self.anchor_sizes,
                    self.anchor_ratios,
                    self.anchor_steps,
                    self.anchor_offset,
                    dtype
                        )
    
    def bboxes_encode(self,labels,bboxes,anchors,scope=None):
        """""
        encode labels and bounding boxes
        """""
        return ssd_common.tf_ssd_bboxes_encode(
            labels,bboxes,anchors,
            self.num_classes,
            ignore_threshold=0.5,
            scope=scope
            )
    
    def detected_bboxes(self,predictions,locations,selected_threshold=None,nms_threshold=0.5,clipping_bbox=None,top_k=400,keep_top_k=200):
        rscores,rbboxes=ssd_common.tf.ssd_bboxes_selected(predictions,locations,selected_threshold=selected_threshold,num_classes=self.num_classes)
        rscores,rbboxes= \
                tfe.bboxes_sort(rscores,rbboxes,top_k=top_k)

        rscores,rbboxes= \
                tfe.bboxes_nms_batch(rscores,rbboxes,nms_threshold=nms_threshold,keep_top_k=keep_top_k)

        return rscores,rbboxes







    def ssd_anchors_all_layers(self,image_shape,layers_shape,anchor_sizes,anchor_ratios,anchor_steps,offset=0.5,dtype=np.float32):
        layers_anchors=[]
        for i,s in enumerate(layers_shape):
            anchor_bboxes=self.ssd_anchor_one_layer(image_shape,
                                               s,
                                               anchor_sizes[i],
                                               anchor_ratios[i],
                                               anchor_steps[i],
                                               offset=offset,
                                               dtype=dtype)
            layers_anchors.append(anchor_bboxes)
        return layers_anchors

    def ssd_anchor_one_layer(self,image_shape,feat_shape,sizes,ratios,step,offset=0.5,dtype=np.float32):
        y,x=np.mgrid[0:feat_shape[0],0:feat_shape[1]]
        y=(y.astype(dtype)+offset)*step/image_shape[0]
        x=(x.astype(dtype)+offset)*step/image_shape[1]

        y=np.expand_dims(y,axis=-1)
        x=np.expand_dims(x,axis=-1)

        num_anchors=len(sizes)+len(ratios)

        h=np.zeros((num_anchors,),dtype=dtype)
        w=np.zeros((num_anchors,),dtype=dtype)

        h[0]=sizes[0]/image_shape[0]
        w[0]=sizes[0]/image_shape[1]
        di=1
        if len(sizes)>1:
            h[1]=math.sqrt(sizes[0]*sizes[1])/image_shape[0]
            w[1]=math.sqrt(sizes[0]*sizes[1])/image_shape[1]
        for i,r in enumerate(ratios):
            h[i+di]=sizes[0]/image_shape[0]/math.sqrt(r)
            w[i+di]=sizes[0]/image_shape[1]/math.sqrt(r)

        return y,x,h,w
        

    def ssd_multibox_layer(self,inputs,sizes,ratios=[1],bn_normalization=False):
        net=inputs
        #number of anchors
		#origin len()
        num_anchors=len(sizes)+len(ratios)
        #locations
        num_loc_pred=num_anchors*4
        loc_pred=conv_layer(net,num_loc_pred,[4,4],layer_name='loc_pred')
        loc_pred=tf.reshape(loc_pred,self.tensor_shape(loc_pred,4)[:-1]+[num_anchors,4])

        #classfication
        num_cls_pred=num_anchors*self.num_classes
        cls_pred=conv_layer(net,num_cls_pred,[4,4],layer_name='conv_cls')
        tf.cast(cls_pred,tf.int32)
        cls_pred=tf.reshape(cls_pred,self.tensor_shape(cls_pred,4)[:-1]+[num_anchors,4])

        return cls_pred,loc_pred


    def densenet_ssd(self,input_x):
        end_points={}
        x=conv_layer(input_x,_filter=2*self.filters,kernel=[7,7],stride=2,layer_name='conv0')

        x=self.dense_block(input_x=x,nb_layers=6,layer_name='dense_1')
        x=self.transition_layer(x,scope='trans_1')
        end_points['block1']=x

        x=self.dense_block(input_x=x,nb_layers=12,layer_name='dense_2')
        x=self.transition_layer(x,scope='trans_2')
        end_points['block2']=x
        
        x=self.dense_block(input_x=x,nb_layers=48,layer_name='dense_3')
        x=self.transition_layer(x,scope='trans_3')
        end_points['block3']=x

        x=self.dense_block(input_x=x,nb_layers=32,layer_name='dense_4')
        x=self.transition_layer(x,scope='trans_4')
        end_points['block4']=x
        
        locations=[]
        predictions=[]
        for i,layer in enumerate(self.feat_layers):
            with tf.variable_scope(layer+'_box'):
                p,l=self.ssd_multibox_layer(end_points[layer],
                                     self.anchor_sizes[i],
                                     self.anchor_ratios[i])
            locations.append(l)
            predictions.append(p)

        return predictions,locations 

    def loss(self,
             logits,
             localisations,
             gclasses,
             glocalisations,
             gscores,
             match_threshold=0.5,
             negative_ratio=3.,
             alpha=1.,
             label_smoothing=0.,
             scope=None ):
        with tf.name_scope(scope, 'ssd_losses'):
            lshape = tfe.get_shape(logits[0], 5)
            num_classes = lshape[-1]
            batch_size = lshape[0]
            final_loss=[]	
   	    	# Flatten out all vectors!
            flogits = []
            fgclasses = []
            fgscores = []
            flocalisations = []
            fglocalisations = []
            for i in range(len(logits)):
                flogits.append(tf.reshape(logits[i], [-1, num_classes]))
                #_flogits=tf.reshape(logits[i],[-1,num_classes])
                #tf.cast(_flogits,tf.int32)
                #flogits.append(_flogits)
                fgclasses.append(tf.reshape(gclasses[i], [-1]))
                fgscores.append(tf.reshape(gscores[i], [-1]))
                flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
                fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
   	 # And concat the crap!
            logits = tf.concat(flogits, axis=0)
            gclasses = tf.concat(fgclasses, axis=0)
            gscores = tf.concat(fgscores, axis=0)
            localisations = tf.concat(flocalisations, axis=0)
            glocalisations = tf.concat(fglocalisations, axis=0)
            dtype = logits.dtype
            #dtype=tf.int32
    	    # Compute positive matching mask...
            pmask = gscores > match_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)
	
    	    # Hard negative mining...
            no_classes = tf.cast(pmask, tf.int32)
            predictions = tf.nn.softmax(logits)
            nmask = tf.logical_and(tf.logical_not(pmask),
    	                           gscores > -0.5)
            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask,
    	                       predictions[:, 0],
    	                       1. - fnmask)
            nvalues_flat = tf.reshape(nvalues, [-1])
    	    # Number of negative entries to select.
            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
            n_neg = tf.minimum(n_neg, max_neg_entries)
	
            val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
            max_hard_pred = -val[-1]
    	    # Final negative mask.
            nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
            fnmask = tf.cast(nmask, dtype)
	
    	    # Add cross-entropy loss.
            with tf.name_scope('cross_entropy_pos'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=gclasses)
                batch_size=tf.to_float(batch_size)
                _sum=tf.reduce_sum(loss*fpmask)
                print(_sum)
                print(batch_size)
                loss=tf.div(_sum,batch_size,name='value')
                #loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
                #tf.losses.add_loss(loss)
                final_loss.append(loss)
                loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
                #tf.losses.add_loss(loss)
                final_loss.append(loss)
    	    # Add localization loss: smooth L1, L2, ...
            with tf.name_scope('localization'):
    	        # Weights Tensor: positive mask + random negative.
                weights = tf.expand_dims(alpha * fpmask, axis=-1)
                loss = custom_layers.abs_smooth(localisations - glocalisations)
                loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
                #tf.losses.add_loss(loss)
                final_loss.append(loss)
                final_loss=tf.add_n(final_loss,name="total_loss")
        return final_loss  

