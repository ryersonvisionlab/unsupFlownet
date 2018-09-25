import tensorflow as tf
from components import *

def photoLoss(flow,downsampledFrame0,downsampledFrame1,alpha,beta):
	with tf.variable_scope(None,default_name="photoLoss"):
		flowShape = flow.get_shape()
		batchSize = flowShape[0]
		height = flowShape[1]
		width = flowShape[2]
		outshape = tf.stack([height,width])

		warpedFrame2 = flowWarp(downsampledFrame1,flow)

		# photometric subtraction
		photoDiff = downsampledFrame0 - warpedFrame2
		photoDist = tf.reduce_sum(tf.abs(photoDiff),axis=3,keep_dims=True)
		robustLoss = charbonnierLoss(photoDist,alpha,beta,0.001)

		return robustLoss
