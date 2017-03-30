import tensorflow as tf
import rvlKit as rvl

import numpy as np

from PIL import Image

#readers
root = "/home/jjyu/datasets/flyingChairs/"
dataList = root+"dataLists/train_im0.txt"
imageReader = rvl.dataInput.reader.Png(root,dataList)
imageReader2 = rvl.dataInput.reader.Png(root,dataList)

readers = [imageReader,imageReader2]
processors = [rvl.dataInput.preProcessor.UniqueCrop([382,382])]
#processors = [rvl.dataInput.preProcessor.SharedCrop([382,382],[384,512])]
processors = processors + processors
shapes = [[382,382,3],[382,382,3]]
types = [tf.uint8,tf.uint8]

queuer = rvl.dataInput.DataQueuer(readers,shapes,types,processors,randomFetch=False)

d,d2 = queuer.queue.dequeue_many(2)

#config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True

with tf.Session(config=config) as sess:
	#sess.run(tf.initialize_all_variables())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	queuer.startQueueing(sess)

	r = sess.run([d,d2])

	arr = np.asarray(r[0][0,:,:,:],np.uint8)
	img = Image.fromarray(arr)
	img.show()

	arr = np.asarray(r[1][0,:,:,:],np.uint8)
	img = Image.fromarray(arr)
	img.show()
