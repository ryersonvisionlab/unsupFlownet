import tensorflow as tf
from deconvLayer import *

def flowRefinementConcat(prev,skip,flow):
	with tf.variable_scope(None,default_name="flowRefinementConcat"):
		with tf.variable_scope(None,default_name="upsampleFlow"):
			upsample = deconvLayer(flow,4,2,2)
		return tf.concat([prev,skip,upsample],3)
