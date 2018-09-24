import tensorflow as tf

def borderOcclusionMask(flow):
	with tf.variable_scope(None,default_name="matchableFlowMask"):
		flowShape = flow.get_shape()

		# make grid
		x = range(flowShape[2].value)
		y = range(flowShape[1].value)
		X, Y = tf.meshgrid(x, y)
		X = tf.expand_dims(X,axis=-1)
		Y = tf.expand_dims(Y,axis=-1)
		X = tf.cast(X,tf.float32)
		Y = tf.cast(Y,tf.float32)

		# mask flows that move off image
		grid = tf.concat([X,Y], -1)
		grid = tf.expand_dims(grid, 0)
		grid = tf.tile(grid, [flowShape[0].value,1,1,1])
		flowPoints = grid + flow
		flowPointsU = tf.expand_dims(flowPoints[:,:,:,0],-1)
		flowPointsV = tf.expand_dims(flowPoints[:,:,:,1],-1)
		mask1 = tf.greater(flowPointsU, 0)
		mask2 = tf.greater(flowPointsV, 0)
		mask3 = tf.less(flowPointsU, flowShape[2].value-1)
		mask4 = tf.less(flowPointsV, flowShape[1].value-1)
		mask = tf.logical_and(mask1, mask2)
		mask = tf.logical_and(mask, mask3)
		mask = tf.logical_and(mask, mask4)
		mask = tf.cast(mask, tf.float32)

		return mask
