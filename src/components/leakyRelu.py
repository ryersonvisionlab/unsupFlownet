import tensorflow as tf

def leakyRelu(x,alpha):
	with tf.variable_scope(None,default_name="leakyRelu"):
		return tf.maximum(x,x*alpha)
