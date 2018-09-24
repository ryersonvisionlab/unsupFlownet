import tensorflow as tf
from leakyRelu import *
from deconvLayer import *

def deconvLayerRelu(x,kernelSize,outMaps,stride):
	with tf.variable_scope(None,default_name="reluDeconv"):
		return leakyRelu(deconvLayer(x,kernelSize,outMaps,stride),0.1)
