import tensorflow as tf

def validPixelMask(lossShape, borderPercentH,borderPercentW):
	with tf.variable_scope(None,default_name="validPixelMask"):
		batchSize = tf.cast(lossShape[0],tf.int32)
		height = tf.cast(lossShape[1],tf.int32)
		width = tf.cast(lossShape[2],tf.int32)
		channels = tf.cast(lossShape[3],tf.int32)

		smallestDimension = tf.minimum(height,width)
		borderThicknessH = tf.cast(tf.round(borderPercentH*tf.cast(height,tf.float32)),tf.int32)
		borderThicknessW = tf.cast(tf.round(borderPercentW*tf.cast(width,tf.float32)),tf.int32)

		innerHeight = height - 2*borderThicknessH
		innerWidth = width - 2*borderThicknessW

		topBottom = tf.zeros(tf.stack([batchSize,borderThicknessH,innerWidth,channels]))
		leftRight = tf.zeros(tf.stack([batchSize,height,borderThicknessW,channels]))
		center = tf.ones(tf.stack([batchSize,innerHeight,innerWidth,channels]))

		mask = tf.concat([topBottom, center, topBottom],1)
		mask = tf.concat([leftRight, mask, leftRight],2)

		#set shape
		ref = tf.zeros(lossShape)
		mask.set_shape(ref.get_shape())

		return mask
