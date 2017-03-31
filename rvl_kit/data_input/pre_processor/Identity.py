from DataPreProcessor import *

class Identity(DataPreProcessor):
	def __init__(self):
		pass
	def attachGraph(self,data_in):
		return data_in
