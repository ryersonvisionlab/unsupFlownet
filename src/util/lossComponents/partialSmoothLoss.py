import tensorflow as tf
from components import *
from smoothLoss import *

def partialSmoothLoss(flow,alpha,beta,mask):
	'''
	blocks gradient to valid pixels marked by mask
	'''
	with tf.variable_scope(None,default_name="partialSmoothLoss"):
		flowValid = flow*mask
		flowInvalid = flow*(1.0-mask)

		# block gradients to valid flow
		flowValid = tf.stop_gradient(flowValid)

		finalFlow = flowValid + flowInvalid

		smooth = smoothLoss(finalFlow,alpha,beta)
		return smooth
