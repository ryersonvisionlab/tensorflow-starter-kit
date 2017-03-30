import tensorflow as tf
import numpy as np

class DataReader(object):
	"""
	reads in data according to a datalist of paths
	handles reading of datalist
	default: creates input graph structure in TF, feed data is only path to data file
	getFeedDict can be overriden to add custom processing on the CPU
	"""
	def __init__(self,datasetRoot,dataListPath):
		self.datasetRoot = datasetRoot
		self.dataListPath = dataListPath

		#read in image list
		with open(dataListPath) as f:
			self.dataListPath = [x[:-1] for x in f.readlines()]

		self.nData = len(self.dataListPath)

	def getFeedDict(self,index):
		path = 	self.datasetRoot + self.dataListPath[index]
		return {self.dataPath: path}
