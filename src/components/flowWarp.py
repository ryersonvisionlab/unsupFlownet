import tensorflow as tf
from flowTransformGrid import *

def flowWarp(data,flow):
	with tf.variable_scope(None, default_name='flowWarp'):
		resampleGrid = flowTransformGrid(flow)
		warped = tf.contrib.resampler.resampler(data,resampleGrid)
		return warped
