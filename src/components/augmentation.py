import tensorflow as tf

def photoAug(image,augParams):
	'''
	constrast (multiplicative brightness)
	additive brightness
	multiplicative color change
	gamma
	gaussian noise

	expects image values from [0,1]
	'''
	with tf.variable_scope(None,default_name="photoAug"):
		#generate random values
		contrast = augParams[0]
		brightness = augParams[1]
		color = augParams[2]
		gamma = augParams[3]
		noiseStd = augParams[4]

		noise = tf.random_normal(image.get_shape(),0,noiseStd)

		#transform
		image = image*contrast + brightness
		image = image*color
		image = tf.maximum(tf.minimum(image,1),0) # clamp between 0 and 1
		image = tf.pow(image,1/gamma)
		image = image+noise

		return image

def photoAugParam(batchSize,contrastMin,contrastMax,brightnessStd,colorMin,colorMax,gammaMin,gammaMax,noiseStd):
	with tf.variable_scope(None,default_name="photoAugParam"):
		contrast = tf.random_uniform([batchSize,1,1,1],contrastMin,contrastMax)
		brightness = tf.random_normal([batchSize,1,1,1],0,brightnessStd)
		color = tf.random_uniform([batchSize,1,1,3],colorMin,colorMax)
		gamma = tf.random_uniform([batchSize,1,1,1],gammaMin,gammaMax)

		noise = noiseStd

		return [contrast,brightness,color,gamma,noise]
