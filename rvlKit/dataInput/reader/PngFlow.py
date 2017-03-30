from DataReader import *

class PngFlow(DataReader):
	def __init__(self,datasetRoot,dataListPath):
		with tf.variable_scope(None,default_name="PngFlowDataReader"):
			DataReader.__init__(self,datasetRoot,dataListPath)

			#graph setup
			flowPath = tf.placeholder(dtype=tf.string)
			png = tf.image.decode_png(tf.read_file(flowPath),dtype=tf.uint16, channels=3)
			png = tf.cast(png,tf.float32)
			flowPng = png[:,:,0:2]
			flowMaskPng = tf.expand_dims(tf.cast(tf.greater(png[:,:,2],0),tf.float32),2)

			#get h,w
			flowShape = tf.shape(png)
			h = flowShape[0]
			w = flowShape[1]

			#convert flow
			flowScale = tf.cast([[[h,w]]],tf.float32)
			flow = ((2.0/(2**16-1))*flowPng) - 1
			flow *= flowScale
			flow *= flowMaskPng

			#expose tensors
			self.dataPath = flowPath
			self.dataOut = flow
