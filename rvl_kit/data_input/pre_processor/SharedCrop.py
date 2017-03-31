from DataPreProcessor import *

class SharedCrop(DataPreProcessor):
	'''
	crops every image it processes the same
	every image must be the same dimension
	expects 3d tensor, single image
	'''
	def __init__(self,crop_shape,in_shape,distribution="uniform"):
		_VALID_DISTRIBUTIONS = ["uniform"]

		assert distribution in _VALID_DISTRIBUTIONS, "Invalid distribution specified"
		self.distribution = distribution # only uniform for now!

		assert len(crop_shape) == 2
		self.crop_h = crop_shape[0]
		self.crop_w = crop_shape[1]

		assert len(in_shape) == 2
		self.in_h = in_shape[0]
		self.in_w = in_shape[1]

		#create shared offset values
		with tf.variable_scope(None,default_name="random_crop_pos"):
			max_h_offset = self.in_h - self.crop_h
			max_w_offset = self.in_w - self.crop_w

			#generate crop, unifrom
			rand_h = tf.random_uniform([],0,max_h_offset,dtype=tf.int32)
			rand_w = tf.random_uniform([],0,max_w_offset,dtype=tf.int32)

		#expose tensors
		self.rand_h = rand_h
		self.rand_w = rand_w

	def attach_graph(self,dataIn):
		with tf.variable_scope(None,default_name="shared_crop"):
			out = tf.slice(dataIn,[self.rand_h,self.rand_w,0],[self.crop_h,self.crop_w,-1])

			return out
