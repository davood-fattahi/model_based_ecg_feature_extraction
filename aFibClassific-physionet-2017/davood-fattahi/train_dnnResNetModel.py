# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:40:31 2022

@author: Davood Fattahi
"""
import tflearn
import tensorflow as tf
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import dropout


"""
Shenda et. al.
https://www.cinc.org/archives/2017/pdf/178-245.pdf

"""


# Building Residual Network
net = tflearn.input_data(shape=[None, n_dim, 1])
# reshape for sub_seq
net = tf.reshape(net, [-1, n_split, 1])
net = tflearn.conv_1d(net, 64, 16, 2)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')

# Residual blocks
net = tflearn.residual_bottleneck(
    net, 2, 16, 64, downsample_strides=2, downsample=True, is_first_block=True)
net = tflearn.residual_bottleneck(
    net, 2, 16, 64, downsample_strides=2, downsample=True)
net = tflearn.residual_bottleneck(
    net, 2, 16, 128, downsample_strides=2, downsample=True)
net = tflearn.residual_bottleneck(
    net, 2, 16, 128, downsample_strides=2, downsample=True)
net = tflearn.residual_bottleneck(
    net, 2, 16, 256, downsample_strides=2, downsample=True)
net = tflearn.residual_bottleneck(
    net, 2, 16, 256, downsample_strides=2, downsample=True)
net = tflearn.residual_bottleneck(
    net, 2, 16, 512, downsample_strides=2, downsample=True)
net = tflearn.residual_bottleneck(
    net, 2, 16, 512, downsample_strides=2, downsample=True)

net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
#net = tflearn.global_avg_pool(net)
# LSTM
# reshape for sub_seq
net = tf.reshape(net, [-1, n_dim//n_split, 512])
net = bidirectional_rnn(net, BasicLSTMCell(256), BasicLSTMCell(256))
net = dropout(net, 0.5)

# Regression
feature_layer = tflearn.fully_connected(net, 32, activation='sigmoid')
net = tflearn.dropout(feature_layer, 0.5)
net = tflearn.fully_connected(net, 4, activation='softmax')
net = tflearn.regression(net, optimizer='adam',  # momentum',
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet',
                    max_checkpoints=10, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, validation_set=(testX, testY),
          show_metric=True, batch_size=300, run_id='resnet', snapshot_step=10,
          snapshot_epoch=False)
