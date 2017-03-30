from DataPreProcessor import *

class SharedCrop(DataPreProcessor):
	'''
	crops every image it processes the same
	every image must be the same dimension
	expects 3d tensor, single image
	'''
	def __init__(self,cropShape,inShape,distribution="uniform"):
		validDistributions = ["uniform"]

		assert distribution in validDistributions, "Invalid distribution specified"
		self.distribution = distribution # only uniform for now!

		assert len(cropShape) == 2
		self.cropH = cropShape[0]
		self.cropW = cropShape[1]

		assert len(inShape) == 2
		self.inH = inShape[0]
		self.inW = inShape[1]

		#create shared offset values
		with tf.variable_scope(None,default_name="randomCropPos"):
			maxHOffset = self.inH - self.cropH
			maxWOffset = self.inW - self.cropW

			#generate crop, unifrom
			randH = tf.random_uniform([],0,maxHOffset,dtype=tf.int32)
			randW = tf.random_uniform([],0,maxWOffset,dtype=tf.int32)

		#expose tensors
		self.randH = randH
		self.randW = randW

	def attachGraph(self,dataIn):
		with tf.variable_scope(None,default_name="SharedCrop"):
			out = tf.slice(dataIn,[self.randH,self.randW,0],[self.cropH,self.cropW,-1])

			return out
