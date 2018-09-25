import tensorflow as tf

def epeEval(predicted,truth,mask):
	with tf.variable_scope(None,default_name="epeEvel"):
		#pixelwise epe
		difference = predicted - truth
		differenceSq = difference * difference
		differenceSqSum = tf.reduce_sum(differenceSq,reduction_indices=[3],keep_dims=True)
		epe = tf.sqrt(differenceSqSum)
		maskedFlow = epe*mask

		#find number of valid pixels
		validPixels = tf.reduce_sum(mask,reduction_indices=[1,2])

		#average epe
		epeSum = tf.reduce_sum(maskedFlow,reduction_indices=[1,2])
		avgEpe = epeSum/validPixels

		return avgEpe, maskedFlow
