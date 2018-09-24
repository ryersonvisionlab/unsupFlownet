import tensorflow as tf
import math

def flowToRgb(flow,zeroFlow="value"):
	with tf.variable_scope(None,default_name="flowToRgb"):
		mag = tf.sqrt(tf.reduce_sum(flow**2,axis=-1))
		ang180 = tf.atan2(flow[:,:,:,1],flow[:,:,:,0])
		ones = tf.ones_like(mag)

		# fix angle so righward motion is red
		ang = ang180*tf.cast(tf.greater_equal(ang180,0),tf.float32)
		ang += (ang180+2*math.pi)*tf.cast(tf.less(ang180,0),tf.float32)

		# normalize for hsv
		largestMag = tf.reduce_max(mag,axis=[1,2])
		magNorm = mag/largestMag
		angNorm = ang/(math.pi*2)

		if zeroFlow == "value":
		        hsv = tf.stack([angNorm,ones,magNorm],axis=-1)
		elif zeroFlow == "saturation":
		        hsv = tf.stack([angNorm,magNorm,ones],axis=-1)
		else:
		        assert("zeroFlow mode must be {'value','saturation'}")
		rgb = tf.image.hsv_to_rgb(hsv)
		return rgb
