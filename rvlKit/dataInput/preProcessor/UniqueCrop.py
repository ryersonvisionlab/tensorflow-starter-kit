from DataPreProcessor import *

class UniqueCrop(DataPreProcessor):
	'''
	crops every image it processes differently
	allows different dimensions for every image
	expects 3d tensor, single image
	'''
	def __init__(self,cropShape,distribution="uniform"):
		validDistributions = ["uniform"]

		assert distribution in validDistributions, "Invalid distribution specified"
		self.distribution = distribution # only uniform for now!

		assert len(cropShape) == 2
		self.h = cropShape[0]
		self.w = cropShape[1]

	def attachGraph(self,dataIn):
		with tf.variable_scope(None,default_name="UniqueCrop"):
			#get crop range
			dataShape = tf.shape(dataIn)
			h = dataShape[0]
			w = dataShape[1]

			maxHOffset = h - self.h
			maxWOffset = w - self.w

			#generate crop, unifrom
			randH = tf.random_uniform([],0,maxHOffset,dtype=tf.int32)
			randW = tf.random_uniform([],0,maxWOffset,dtype=tf.int32)

			out = tf.slice(dataIn,[randH,randW,0],[self.h,self.w,-1])

			return out
