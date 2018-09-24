import tensorflow as tf

def charbonnierLoss(x,alpha,beta,epsilon):
	with tf.variable_scope(None,default_name="charbonnierLoss"):
		epsilonSq = epsilon*epsilon
		xScale = x*beta

		return tf.pow(xScale*xScale + epsilonSq,alpha)
