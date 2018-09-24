import tensorflow as tf
from convLayerRelu import *

def resSkip(x,outMaps,stride):
	"""
	skip component for res layer for when the resolution changes during stride
	number of channels also changes via linear combination
	"""
	with tf.variable_scope(None, default_name="resSkip"):
		return convLayer(x,1,outMaps,stride)

def resLayer(x,kernelSize,outMaps):
	"""
	a reslayer, no striding
	"""
	with tf.variable_scope(None, default_name="resLayer"):
		skip = x
		conv1 = convLayerRelu(x,kernelSize,outMaps,1)
		conv2 = convLayer(conv1,kernelSize,outMaps,1)
		return leakyRelu(conv2 + skip, 0.1)

def resLayerStride(x,kernelSize,outMaps):
	"""
	a reslayer, stride by 2
	"""
	with tf.variable_scope(None, default_name="resLayerStride"):
		skip = resSkip(x,outMaps,2)
		conv1 = convLayerRelu(x,kernelSize,outMaps,2)
		conv2 = convLayer(conv1,kernelSize,outMaps,1)
		return leakyRelu(conv2 + skip, 0.1)

def resBlock(x,kernelSize,outMaps):
	"""
	block of resnet units, resolution halves
	"""
	with tf.variable_scope(None, default_name="resBlock"):
		res1 = resLayerStride(x,kernelSize,outMaps)
		res2 = resLayer(res1,kernelSize,outMaps)
		return res2
