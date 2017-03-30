from DataReader import *

class Png(DataReader):
	"""
	Manages the fetching of single peices of data on the cpu onto the gpu
	"""
	def __init__(self,datasetRoot,dataListPath,channels=3,dtype=tf.uint8):
		with tf.variable_scope(None,default_name="ImageDataReader"):
			DataReader.__init__(self,datasetRoot,dataListPath)

			#graph setup
			imgPath = tf.placeholder(dtype=tf.string)
			img = tf.image.decode_png(tf.read_file(imgPath), channels=channels, dtype=dtype)

			#expose tensors
			self.dataPath = imgPath
			self.dataOut = img
