import tensorflow as tf

def batchMeshGrid(batchSize, height, width):
	with tf.variable_scope(None, default_name='batchMeshGrid'):
		# make grid
		x = range(width)
		y = range(height)
		X, Y = tf.meshgrid(x, y)
		X = tf.cast(X,tf.float32)
		Y = tf.cast(Y,tf.float32)
		grid = tf.stack([X,Y], axis=-1)

		# tile for batch
		batchGrid = tf.expand_dims(grid, 0)
		batchGrid = tf.tile(batchGrid, [batchSize,1,1,1])

		return batchGrid

def batchMeshGridLike(tensor):
	with tf.variable_scope(None, default_name='batchMeshGridLike'):
		shape = tensor.get_shape()
		return batchMeshGrid(shape[0].value,shape[1].value,shape[2].value)
