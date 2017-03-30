from DataPreProcessor import *

class Identity(DataPreProcessor):
	def __init__(self):
		pass
	def attachGraph(self,dataIn):
		return dataIn
