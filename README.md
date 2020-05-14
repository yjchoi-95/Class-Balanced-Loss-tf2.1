# Class-Balanced-Loss-tf2.1
[CVPR 2019] Class-Balanced Loss Based on Effective Number of Samples

#### Example
```
from utils import *
import tensorflow as tf

## load cifar 10
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

## change labels shape
train_labels_ = tf.one_hot(train_labels, depth=10).numpy().reshape(-1, 10)
test_labels_ = tf.one_hot(test_labels, depth=10).numpy().reshape(-1, 10)

## calculate # of samples, beta, class balanced term per class
CB_ = CB_utils(train_labels_im)
n_dict, beta_dict, cb_dict = CB_.get_results()

## model compile
cb_vgg16 = vgg16_base(shape_ = train_images.shape[1:])
cb_vgg16.compile(optimizer=tf.keras.optimizers.Adam(), loss = CB_loss(cb_dict), metrics = ['accuracy'])

## model fitting with class-balanced-loss
history_cb = cb_vgg16.fit(train_images_im, train_labels_im_, validation_split=0.1, epochs = 200, batch_size = 256)


```


