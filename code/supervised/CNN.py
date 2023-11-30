#import gym
#from RL_brain import DeepQNetwork

import tensorflow as tf
import numpy as np


class CNN(object):
	"""docstring for network"""
	def __init__(self, 
		input_size, # 2 dimension.
		output_size,
		channels = 1,
		learning_rate = 0.05,
		lr_decay_step = 15000,
		lr_decay_rate = 0.9,
		net_name = 'cnn',
		):
		#self.learning_rate = learning_rate
		self.global_step = tf.Variable(0, trainable=False)
		self.lr = tf.train.exponential_decay(learning_rate, self.global_step, lr_decay_step, lr_decay_rate, staircase=True)
		
		self.size = self.cnn_size(input_size=input_size, output_size=output_size, channels=channels)
		self.build_net(net_name = net_name)
		self.saver = tf.train.Saver()

		#correct_pred = tf.equal(tf.cast(tf.argmax(self.pred,axis=1),"float"), self.labels)
		correct_pred = tf.equal(tf.argmax(self.pred,axis=1), tf.argmax(self.labels,axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred,"float"))
		
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def build_net(self, net_name):
		
		self.input = tf.placeholder(tf.float32,  self.size.input_size, name='input')  # input
		self.labels = tf.placeholder(tf.float32, [None, self.size.output_size], name='label')  # label
		self.pred = self.build_cnn(self.input, net_name)
		#onehot_labels = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=self.size.output_size)
		onehot_labels = self.labels
		self.loss = tf.reduce_mean(tf.squared_difference(self.pred, onehot_labels))
		self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
	def build_cnn(self, inputs, net_name, activation_function=tf.nn.softmax, init_range = 0.001):
		w_initializer = tf.random_uniform_initializer(-init_range, init_range)
		conv = inputs
		with tf.variable_scope(net_name):
			for j in range(self.size.cnn_layers):
				conv = tf.layers.conv2d(
					inputs=conv,
					filters=32,
					strides = 2,
					kernel_size=self.size.kernel_size,
					padding="same",
					activation=tf.nn.relu)
			flat = tf.contrib.layers.flatten(conv)
			dense = flat
			for j in range(self.size.dense_layers):
				dense = tf.layers.dense(
					inputs=dense, 
					units=self.size.dense_size, 
					kernel_initializer=w_initializer, 
					activation = tf.nn.relu)
			output = tf.layers.dense(
					inputs=dense, 
					units=self.size.output_size, 
					kernel_initializer=w_initializer, 
					name='output', 
					activation = activation_function)
		return output
	def train(self, inputs, labels):
		self.sess.run(self._train_op, feed_dict = {self.input: inputs, self.labels: labels})

	def test(self, inputs, labels):
		return self.sess.run(self.accuracy, feed_dict = {self.input: inputs, self.labels: labels})
	def predict(self, inputs):
		#return self.sess.run(self.pred, feed_dict = {self.input: inputs})
		# For classification problem
		return self.sess.run(tf.argmax(self.pred,axis=1), feed_dict = {self.input: inputs})
	
	def show_loss(self, inputs, label):
		return self.sess.run(self.loss, feed_dict = {self.input: inputs, self.labels: label})
	def get_lr(self):
		return self.sess.run(self.lr)
	def train_step(self):
		return self.sess.run(self.global_step)
	def reset_train_step(self):
		self.sess.run(self.global_step.assign(0))

	def save(self, filename):
		self.saver.save(self.sess, filename)
	def restore(self, filename):
		self.saver.restore(self.sess, filename)

	class cnn_size():
		def __init__(self,
			input_size,
			output_size,
			channels = 1,
			cnn_layers = 2,
			kernel_size = [5,5],
			dense_layers = 1,
			dense_size = 1024,
			):
			self.channels = channels
			self.input_size = [None, input_size[0], input_size[1], channels]
			self.kernel_size = kernel_size
			self.cnn_layers = cnn_layers
			self.dense_layers = dense_layers
			self.dense_size = dense_size
			self.output_size = output_size



if __name__ == '__main__':
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	test_inputs = mnist.test.images
	test_inputs = np.reshape(test_inputs, [-1, 28, 28, 1])
	test_labels = mnist.test.labels

	#my_size = cnn_size(input_size=[28,28], output_size = 10)
	#my_cnn = CNN(cnn_size=my_size, learning_rate=0.001)
	my_cnn = CNN(input_size=[28,28], output_size = 10, learning_rate=0.001)

	#my_cnn.restore('test.ckpt')

	from pdb import set_trace
	#set_trace()
	for step in range(5000):
		train_inputs, train_labels = mnist.train.next_batch(100)
		train_inputs = np.reshape(train_inputs, [-1, 28, 28, 1])
		my_cnn.train(inputs = train_inputs, labels = train_labels)
		if step %100 ==0:
			test_accu = my_cnn.test(inputs = test_inputs, labels = test_labels)
			print('Step {}. Test accuracy {:.2f}%'.format(step, test_accu*100))