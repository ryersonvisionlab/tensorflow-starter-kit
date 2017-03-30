#!/usr/bin/python

import tensorflow as tf
import numpy as np
import threading
from scipy import misc
from random import randint
#from components import *
import itertools
import preProcessor

class DataQueuer(object):
	"""
	Manages the concurrent fetching of data on the cpu onto the gpu
	does not guarantee that images are appended in order
	"""

	def __init__(self,dataReaders, dataShapes, dataTypes, preProcessors=[],nThreads=1,randomFetch = True, randomCrop=False):
		self.nThreads = nThreads
		self.randomFetch = randomFetch
		self.inputHeight = 1024
		self.inputWidth = 2048
		self.dataReaders = dataReaders
		if randomCrop:
			self.desiredHeight = 512
			self.desiredWidth = 1024
		else:
			self.desiredHeight = self.inputHeight
			self.desiredWidth = self.inputWidth

		with tf.variable_scope(None,default_name="queuer"):
			#ensure all readers have the same number of data
			self.nData = dataReaders[0].nData
			self.next_data = 0
			for reader in dataReaders:
				assert self.nData == reader.nData, "readers do not read the same number of data"
			assert len(dataShapes) == len(dataReaders), "# of dataShapes must equal dataReaders"
			assert len(dataTypes) == len(dataReaders), "# of dataTypes must equal dataReaders"

			#if no preProcessors given, replace all with identity processors
			if len(preProcessors) == 0:
				for it,reader in enumerate(dataReaders):
					preProcessors.append(preProcessor.Identity())

			assert len(dataReaders) == len(preProcessors), "number of preProcessors not equal number of data readers"

			#thread lock
			self.lock = threading.Lock()

			#get outputs from readers
			dataOutputs = []
			#dataShapes = []
			#dataTypes = []
			for it,reader in enumerate(dataReaders):
				#pass reader through proprocessor
				processor = preProcessors[it]
				data = processor.attachGraph(reader.dataOut)
				dataOutputs.append(data)

			#queue
			self.queue = tf.FIFOQueue(shapes=dataShapes, dtypes=dataTypes, capacity=nThreads*8)
			self.enqueue_op = self.queue.enqueue(dataOutputs)

	def getNextDataIndex(self):
		with self.lock:
			index = self.next_data
			self.next_data += 1

			if self.next_data >= self.nData:
				self.next_data = 0

			return index

	def getNextDataIndexRandom(self):
		with self.lock:
			index = randint(0,self.nData-1)

			return index

	def thread_main(self, sess):
		while True:
			if self.randomFetch:
				index = self.getNextDataIndexRandom()
			else:
				index = self.getNextDataIndex()

			#get feed dicts from readers
			feedDict = {}
			for reader in self.dataReaders:
				feedDict.update(reader.getFeedDict(index))

			sess.run([self.enqueue_op],feed_dict=feedDict)

	def startQueueing(self,sess):
		for i in range(self.nThreads):
			thread = threading.Thread(target=self.thread_main, args=(sess,))
			thread.daemon = True
			thread.start()
