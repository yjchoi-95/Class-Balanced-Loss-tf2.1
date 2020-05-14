#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import Counter

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import math
import time

def im_factor_(train_labels, idx) :
    return np.min([x for x in Counter(pd.DataFrame(train_labels[idx,:])[0]).values()])

def make_imbalanced_set(train_images, train_labels, major_class, minor_class, imbalance_factor_ = 20, seed_num = 0) :
    imbalance_ratio = 1 / imbalance_factor_
    ## major index
    major_idx = np.where(np.squeeze(train_labels) == major_class)[0]

    ## main minor index
    temp_minor_idx = np.where(np.squeeze(train_labels) == minor_class)[0]
    np.random.seed(seed_num)
    main_minor_idx = np.random.choice(temp_minor_idx, size = int(5000 * imbalance_ratio), replace = False)
    
    ## minor index
    other_minor_idx = np.where(np.squeeze(train_labels) != major_class)[0]
    other_minor_idx = other_minor_idx[np.where(np.squeeze(train_labels[other_minor_idx,:]) != minor_class)[0]]
    np.random.seed(seed_num)
    minor_idx = np.random.choice(other_minor_idx, size = int(40000 * np.random.uniform(low = imbalance_ratio + 0.01)) , replace = False)
    
    ## total index
    total_idx = np.hstack([major_idx, main_minor_idx, minor_idx])
    np.random.seed(seed_num)
    total_idx = np.random.choice(total_idx, size = total_idx.shape[0], replace = False)
    return train_images[total_idx,:,:,:], train_labels[total_idx,:]

def vgg16_base(shape_) :
    
    input_layer = tf.keras.Input(shape = shape_)
    
    ## first conv block
    conv_x = tf.keras.layers.Conv2D(4, (3,3), activation = "relu", padding = "same")(input_layer)
    conv_x = tf.keras.layers.Conv2D(4, (3,3), activation = "relu", padding = "same")(conv_x)
    pool_x = tf.keras.layers.MaxPool2D((2,2), strides=(2,2))(conv_x)
    
    ## second conv block
    conv_x = tf.keras.layers.Conv2D(8, (3,3), activation = "relu", padding = "same")(pool_x)
    conv_x = tf.keras.layers.Conv2D(8, (3,3), activation = "relu", padding = "same")(conv_x)
    pool_x = tf.keras.layers.MaxPool2D((2,2), strides=(2,2))(conv_x)

    ## third conv block
    conv_x = tf.keras.layers.Conv2D(16, (3,3), activation = "relu", padding = "same")(pool_x)
    conv_x = tf.keras.layers.Conv2D(16, (3,3), activation = "relu", padding = "same")(conv_x)
    conv_x = tf.keras.layers.Conv2D(16, (3,3), activation = "relu", padding = "same")(conv_x)
    pool_x = tf.keras.layers.MaxPool2D((2,2), strides=(1,1))(conv_x)

    ## forth conv block
    conv_x = tf.keras.layers.Conv2D(32, (3,3), activation = "relu", padding = "same")(pool_x)
    conv_x = tf.keras.layers.Conv2D(32, (3,3), activation = "relu", padding = "same")(conv_x)
    conv_x = tf.keras.layers.Conv2D(32, (3,3), activation = "relu", padding = "same")(conv_x)
    pool_x = tf.keras.layers.MaxPool2D((2,2), strides=(1,1))(conv_x)

    ## fifth conv block
    conv_x = tf.keras.layers.Conv2D(32, (3,3), activation = "relu", padding = "same")(pool_x)
    conv_x = tf.keras.layers.Conv2D(32, (3,3), activation = "relu", padding = "same")(conv_x)
    conv_x = tf.keras.layers.Conv2D(32, (3,3), activation = "relu", padding = "same")(conv_x)
    pool_x = tf.keras.layers.MaxPool2D((2,2), strides=(1,1))(conv_x)
    
    ## fc layer
    flatten_x = tf.keras.layers.Flatten()(pool_x)
    dense_x = tf.keras.layers.Dense(128, activation = "relu")(flatten_x)
    dense_x = tf.keras.layers.Dense(128, activation = "relu")(dense_x)
    output_x = tf.keras.layers.Dense(10, activation = "softmax")(dense_x)
    
    vgg16 = tf.keras.Model(inputs = input_layer, outputs = output_x)
    return vgg16

class CB_utils :
    def __init__(self, labels) :
        self.data = labels
        self.pd_data = pd.DataFrame(labels)[0]
        
    def beta_(self, labels) :
        n_ = dict(Counter(labels))
        beta_ = {beta_i:((n_i - 1) / n_i) for beta_i, n_i in n_.items()}
        return n_, beta_
    
    def class_balanced_term(self, beta, n) :
        return (1 - beta) / (1 - (beta ** n))
    
    def make_cb_dict(self, n_dict, beta_dict) :
        self.temp_cb_dict = {cb_i[0]:self.class_balanced_term(cb_i[1], n_i) for cb_i, n_i in zip(beta_dict.items(), n_dict.values())}
        
        class_i = [x for x in self.temp_cb_dict.keys()]
        cb_i = [x for x in self.temp_cb_dict.values()]
        
        # build a lookup table
        cb_dict = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(class_i),
                                                                                                 values=tf.constant(cb_i)),
                                                 default_value=tf.constant(-1.),
                                                 name="class_weight")
        return cb_dict

    def get_results(self) :
        n_dict, beta_dict = self.beta_(self.pd_data)
        cb_dict = self.make_cb_dict(n_dict, beta_dict)
        return n_dict, beta_dict, cb_dict

def CB_loss(cb_dict):
    origin_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    def inter_loss(y_true, y_pred):
        ## calculate original loss
        cce_ = origin_loss(y_true, y_pred)
        
        ## change dtype for use lookup table
        y_key = tf.math.argmax(y_true, axis = 1, output_type = tf.int32)
        
        ## adjust cb term 
        cb_term = cb_dict.lookup(y_key)
        cb_cce_ = tf.math.multiply(cb_term, cce_)
        return tf.keras.backend.sum(cb_cce_)
    return inter_loss

