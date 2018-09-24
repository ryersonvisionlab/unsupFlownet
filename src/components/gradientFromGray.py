import tensorflow as tf

def gradientFromGray(img):
	with tf.variable_scope(None,default_name="forwardDifferencesSingle"):
		kernel = tf.transpose(tf.constant([\
			[ \
				[ \
					[0,0,0],\
					[0,1,-1],\
					[0,0,0]\
				] \
			], \
			[ \
				[ \
					[0,0,0],\
					[0,1,0],\
					[0,-1,0]\
				] \
			] \
		],dtype=tf.float32),perm=[3,2,1,0])

		return tf.nn.conv2d(img,kernel,[1,1,1,1],padding="SAME")
