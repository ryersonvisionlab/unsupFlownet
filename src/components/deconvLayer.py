import tensorflow as tf

def deconvLayer(x,kernelSize,outMaps,stride): #default caffe style MRSA
	with tf.variable_scope(None,default_name="deconv"):
		inMaps = x.get_shape()[3]

		kShape = [kernelSize,kernelSize,outMaps,inMaps]
		w = tf.get_variable("weights",shape=kShape,initializer=tf.uniform_unit_scaling_initializer())
		tf.add_to_collection("weights",w)

		inshape = x.get_shape()
		outShape = tf.stack([inshape[0],inshape[1]*stride,inshape[2]*stride,outMaps],name="shapeEval")

		deconv = tf.nn.conv2d_transpose(x,w,outShape,[1,stride,stride,1],padding="SAME")
		return deconv
