import tensorflow as tf

def convLayer(x,kernelSize, outMaps, stride): #default caffe style MRSA
	with tf.variable_scope(None,default_name="conv"):
		inMaps = x.get_shape()[3]

		kShape = [kernelSize,kernelSize,inMaps,outMaps]
		w = tf.get_variable("weights",shape=kShape,initializer=tf.uniform_unit_scaling_initializer())
		b = tf.get_variable("biases",shape=[outMaps],initializer=tf.constant_initializer(0))

		tf.add_to_collection("weights",w)
		conv = tf.nn.conv2d(x,w,strides=[1,stride,stride,1], padding="SAME",name="conv2d")
		return conv + b
