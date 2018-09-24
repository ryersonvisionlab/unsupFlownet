import tensorflow as tf
from batchMeshGrid import *

def flowTransformGrid(flow):
	with tf.variable_scope(None,default_name="flowTransformGrid"):
		resampleGrid = batchMeshGridLike(flow) + flow
		return resampleGrid
