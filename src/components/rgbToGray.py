import tensorflow as tf

def rgbToGray(img):
	with tf.variable_scope(None,default_name="rgbToGray"):
		rgbWeights = [0.2989,0.5870,0.1140]
		rgbWeights = tf.expand_dims(tf.expand_dims(tf.expand_dims(rgbWeights,0),0),0)

		weightedImg = img*rgbWeights

		return tf.reduce_sum(weightedImg,reduction_indices=[3],keep_dims=True)
