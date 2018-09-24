import tensorflow as tf
from leakyRelu import *
from convLayer import *

def convLayerRelu(x,kernelSize,outMaps,stride):
	with tf.variable_scope(None,default_name="reluConv"):
		return leakyRelu(convLayer(x,kernelSize,outMaps,stride),0.1)
