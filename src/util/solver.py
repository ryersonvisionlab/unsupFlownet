#!/usr/bin/python

import tensorflow as tf
from components import *

#should move this to hyper params
momentum1 = 0.9
momentum2 = 0.999

def attachSolver(loss):
	with tf.variable_scope(None,default_name="solver"):
		learningRate = tf.placeholder(tf.float32,shape=[])
		solver = tf.train.AdamOptimizer(learning_rate=learningRate , beta1=momentum1, beta2=momentum2).minimize(loss)
		return [solver, learningRate]
