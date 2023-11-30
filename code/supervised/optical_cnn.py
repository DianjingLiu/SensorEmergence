# import numpy as np
import tensorflow as tf
import numpy as np
import math
import pdb
import propagator_tool4 as ppg
from CNN import CNN
from pdb import set_trace
tf.reset_default_graph()
class optical_cnn(CNN):
	def __init__(
		self, 
		input_size, # 2 dimension.
		output_size,
		channels = 1,
		learning_rate = 0.05,
		learning_rate_mod = 0.01,
		lr_decay_step = 15000,
		lr_decay_rate = 0.9,
		net_name = 'cnn',
	):
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			channels=channels,
			learning_rate=learning_rate,
			lr_decay_step=lr_decay_step,
			lr_decay_rate=lr_decay_rate,
			net_name=net_name,
		)
		self.set_training_op(learning_rate_mod)


	def set_training_op(self,learning_rate_mod=None):
		# parse learning rate config
		if learning_rate_mod is  None:
			learning_rate_mod = self.get_lr()
			self.lr_mod = tf.Variable(learning_rate_mod, trainable=False)
		elif type(learning_rate_mod) is float:
			print('constant mod lr', learning_rate_mod)
			self.lr_mod = tf.Variable(learning_rate_mod, trainable=False)
		elif len(learning_rate_mod) == 1:
			print('constant mod lr', learning_rate_mod[0])
			learning_rate_mod=learning_rate_mod[0]
			self.lr_mod = tf.Variable(learning_rate_mod, trainable=False)
		else:
			lr_mod, decay_steps, decay_rate = learning_rate_mod
			self.lr_mod = tf.train.exponential_decay(lr_mod, self.global_step, decay_steps=int(decay_steps), decay_rate=decay_rate, staircase=False)
		# Get CNN and optical layer parameters
		self.cnn_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
		self.mod_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='optical_layer')
		# Set up training operator for CNN and optical layer separately
		with tf.variable_scope('train'):
			increment_global_step_op = tf.assign(self.global_step, self.global_step+1)
			opt1 = tf.train.AdamOptimizer(self.lr_mod, name='Optim_mod')
			#opt1 = tf.train.RMSPropOptimizer(self.lr_mod, name='Optim_mod')
			opt2 = tf.train.AdamOptimizer(self.lr, name="Optim_rl")
			grads = tf.gradients(self.loss, self.mod_params + self.cnn_params)
			grads1 = grads[:len(self.mod_params)]
			grads2 = grads[len(self.mod_params):]
			train_op1 = opt1.apply_gradients(zip(grads1, self.mod_params))
			train_op2 = opt2.apply_gradients(zip(grads2, self.cnn_params))
			self._train_op = tf.group(train_op1, train_op2, increment_global_step_op)
		self.sess.run(tf.global_variables_initializer())
	def build_cnn(self, inputs, name):
		'''
		# single channel inputs: inputs shape is [N, H, W, C=1]
		with tf.variable_scope(name):
			img = self.optical_layer(inputs)
		'''
		# multi-channel inputs: inputs shape is [N, H, W, C]
		imgs = self.prop_multichannel('optical_layer', inputs)
		#imgs=inputs
		self.imgs = imgs
		return super().build_cnn(imgs,name)
	def prop_multichannel(self, name, inputs):
		# Note: the input and output of optical_layer() is assumed to have shape: [None, dim, dim]
		# Adapt optical_layer to process multi-channel inputs
		shape = inputs.shape # [N, H, W, C]
		imgs = tf.transpose(inputs, [0,3,1,2]) # [N, C, H, W]
		imgs = tf.reshape(imgs, (-1, shape[1], shape[2])) # [N * C, H, W]
		with tf.variable_scope(name):
			imgs = self.optical_layer(imgs)
		imgs = tf.reshape(imgs, (-1, shape[3], shape[1], shape[2]))
		imgs = tf.transpose(imgs, [0,2,3,1])
		imgs = imgs * 50
		return imgs
		
	def update_mask(self, dim_hole):
		dim_mod = self.dim_mod  # resolution of modulator
		dim_mod_before_interp = self.dim_mod_before_interp  # resolution of interpolation points on modulator
		mask = ppg.mask_modulator_round(dim_mod, dim_hole)
		self.sess.run(tf.assign(self.mask, mask))
		dim_hole_interp = np.floor(dim_hole / dim_mod * dim_mod_before_interp).astype('int32')
		mask_interp = ppg.mask_modulator_round(dim_mod_before_interp, dim_hole_interp)
		self.sess.run(tf.assign(self.mask_interp, mask_interp))
	def show_modulator(self):
		modulator = self.sess.run(self.modulator_draw)
		modulator = modulator * self.unitchange / 1e-6
		return modulator
	def show_modulator_params(self):
		return self.get_mod_params()
	def update_mod(self, params):
		mod = np.zeros([1,25,1])
		mod[0, [7,11,12,13,17],0] = params
		#set_trace()
		self.sess.run(tf.assign(self.modulator_before_interp, mod))
		#self.sess.run(tf.global_variables_initializer())
	def get_mod_params(self):
		mod = self.sess.run(self.modulator_before_interp)
		return mod
	def show_sensor(self, inputs):
		return self.sess.run(self.imgs, feed_dict={self.input:inputs})
	def optical_layer(self, 
			inputs,
			#dim_mod = 80  # resolution of modulator
			dim_mod = 40,
			dim_mod_before_interp = 10, # resolution of interpolation points on modulator
			dim_hole = 30,   # initial resolution of hole of the mask
			mylambda = 525.e-9,  # wavelength
			focal_length = 15e-3, # unit is meter
			len_modulator = .5e-3, # length of modulator, unit is meter
			len_obj = 1e-3,
			unitchange = 2e-5,#1.e-5,  # change the unit of computation for a better accuracy
			):
		dist_obj = 47.14e-3
		dist_img = 22e-3
		dim_obj = int(inputs.shape[1]) # input shape is [N, H, W] or [N*C, H, W] with H = W
		dim_det = dim_obj
		dim_mod_mask = dim_mod
		len_det = len_obj/2
		self.unitchange = unitchange
		self.dim_mod = dim_mod
		self.dim_mod_before_interp = dim_mod_before_interp
		#self.dim_hole = dim_hole

		# incoherent light propagators
		dim_obj_total = dim_obj * dim_obj
		dim_det_total = dim_det * dim_det
		dim_mod_total = dim_mod * dim_mod

		# propogator matrices
		propagator1_real, propagator1_imag = ppg.propagator_compute(dim_obj, len_obj, dim_mod, len_modulator, dist_obj, mylambda)    
		propagator2_real, propagator2_imag = ppg.propagator_compute(dim_mod, len_modulator, dim_det, len_det, dist_img, mylambda)  
		
		# Variable: thickness of 10*10 points on modulator
		initial_mod_unitchanged = np.zeros([1, dim_mod_before_interp * dim_mod_before_interp, 1]).astype('float32')
		#############################################################################
		'''
		# initialize as a perfect lens
		mod = ppg.lens_initialize(dim_mod_before_interp, focal_length, len_modulator) 
		# if using a hole
		base = ppg.lens_initialize(dim_mod, focal_length, len_modulator) 
		base = np.reshape(base, (dim_mod, dim_mod))
		base = base[dim_mod//2, dim_mod//2 - int(dim_hole*0.707)]
		mod = mod - base
		initial_mod_unitchanged = np.reshape(mod, [1, dim_mod_before_interp * dim_mod_before_interp, 1]).astype('float32') / self.unitchange
		#set_trace()
		#'''
		#############################################################################
		self.modulator_before_interp = tf.Variable(initial_mod_unitchanged, trainable = True, dtype = tf.float32, name = 'modulator')
		#self.modulator_before_interp = tf.Variable(initial_mod_unitchanged, trainable = False, dtype = tf.float32, name = 'modulator')
		# mask on interp_modulator 10*10
		dim_mod_mask_interp = dim_mod_before_interp
		dim_hole_interp = np.floor(dim_hole / dim_mod_mask * dim_mod_mask_interp).astype('int32')
		self.mask_interp = tf.Variable(ppg.mask_modulator_round(dim_mod_mask_interp, dim_hole_interp), trainable = False)
		#self.mask_interp = ppg.mask_modulator_round(dim_mod_mask_interp, dim_hole_interp)

		# adding mask on 10 * 10 points
		modulator_before_interp_mask = self.modulator_before_interp * tf.reshape(self.mask_interp, [1, dim_mod_mask_interp * dim_mod_mask_interp, 1])
		
		# interpolation of modulator to 80*80
		# interpolation using thin-plate-spline method
		initial_mod_lens_interp, train_points, query_points = ppg.lens_initialize_interp(dim_mod_before_interp, dim_mod, focal_length, len_modulator)
		modulator_interp = tf.contrib.image.interpolate_spline(
			train_points = train_points,
			train_values = tf.cast(modulator_before_interp_mask, tf.float64),
			query_points = query_points,
			order = 2,
			regularization_weight=0.0,
			name = 'interpolate_spline'
		)
		modulator_interp = tf.cast(modulator_interp, tf.float32)

		# lens modulator
		self.modulator = tf.reshape(modulator_interp, [dim_mod_total, 1])
		# For debug: set modulator as a lens
		#self.modulator = tf.cast(tf.reshape(ppg.lens_initialize(dim_mod, focal_length, len_modulator), [dim_mod_total, 1]), tf.float32) / self.unitchange
		# phase = thickness of modulator * 2pi / wavelength
		modulator_phase = self.modulator * 2. * math.pi / (mylambda / self.unitchange) # unit of length is 10^-6 m = 1 micron  

		# Calculation of OutputIntensity = P2 * M * P1 * InputIntensity
		# InputIntensity, default phase is 0 on the whole plane
		input_images = tf.reshape(tf.sqrt(inputs), [-1, dim_obj * dim_obj])      # calculate amplitude 
		object_amplitude = tf.transpose(input_images)    # change shape from [batch size, 28 * 28] to [28 * 28, batch size]
		object_random_phase = tf.zeros(tf.shape(object_amplitude)) * 2. * math.pi    # without random phase on input image
		object_real = object_amplitude * tf.cos(object_random_phase)
		object_imag = object_amplitude * tf.sin(object_random_phase)
		object_intensity = (object_real**2. + object_imag**2.) / (mylambda)**2. / (dim_obj * dim_mod)**4.
		# M, with mask on it to modulate amplitude
		self.mask = tf.Variable(ppg.mask_modulator_round(dim_mod_mask, dim_hole), trainable = False)    
		#self.mask = ppg.mask_modulator_round(dim_mod_mask, dim_hole)
		phase_modulation_real = tf.reshape(tf.cos(modulator_phase),[-1, 1]) * tf.reshape(self.mask, [dim_mod_total, 1])
		phase_modulation_imag = tf.reshape(tf.sin(modulator_phase),[-1, 1]) * tf.reshape(self.mask, [dim_mod_total, 1])
		# M * P1
		MP1_real = phase_modulation_real * propagator1_real - phase_modulation_imag * propagator1_imag
		MP1_imag = phase_modulation_imag * propagator1_real + phase_modulation_real * propagator1_imag
		# P2 * M * P1
		P2MP1_real = tf.matmul(propagator2_real, MP1_real) - tf.matmul(propagator2_imag, MP1_imag)
		P2MP1_imag = tf.matmul(propagator2_imag, MP1_real) + tf.matmul(propagator2_real, MP1_imag)
		P2MP1 = P2MP1_real**2. + P2MP1_imag**2.
		# OutputIntensity = |P2 * M * P1| * InputIntensity, for incoherent light
		output_tmpt = tf.matmul(P2MP1, object_intensity) 

		# adding noise on detector
		# white noise 
		noise_rate =  0 # 1.0        # with or without noise
		tfrandom_noise = tf.random_normal(
			shape = tf.shape(output_tmpt), 
			mean=0.0,
			stddev=1.0,
			dtype=tf.float32)
		# white noise is propotional to step
		detector_noise = tf.cast(self.global_step, tf.float32) * noise_rate * tf.maximum(tfrandom_noise, 0.0)
		
		# propotional noise 
		tfrandom_noise_propotional = tf.random_normal(
			shape = tf.shape(output_tmpt),
			mean=0.0,
			stddev=0.2,
			dtype=tf.float32)
		propotional_random_rate = 0 #1.0   # with or without noise
		propotional_random_noise = propotional_random_rate * tfrandom_noise_propotional     # with random noise on input image
		
		# add propotional noise and white noise on the detector
		detector_with_noise = tf.maximum(output_tmpt * (1.0 + propotional_random_noise) + detector_noise, 0.)

		# the intensity on the detector is normalized by L2 normalize
		self.detector_origin = tf.nn.l2_normalize(detector_with_noise, axis = 0) 

		# the output is transformed into shape of [batch size, 28 * 28]
		output_detector = tf.transpose(self.detector_origin)
		
		# some utilities
		# adding phase mask / showing the amplitude mask region


		#self.modulator_draw = tf.reshape(modulator_interp, [dim_mod, dim_mod]) * tf.reshape(self.mask, [dim_mod, dim_mod])
		self.modulator_draw = tf.reshape(self.modulator, [dim_mod, dim_mod]) * tf.reshape(self.mask, [dim_mod, dim_mod])
		#return tf.reshape(output_detector, [-1, dim_det, dim_det, 1])
		return tf.reshape(output_detector, [-1, dim_det, dim_det])
class dummy_env_mnist():
	def __init__(self):
		from tensorflow.examples.tutorials.mnist import input_data
		mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
		test_inputs = mnist.test.images
		test_inputs = np.reshape(test_inputs, [-1, 28, 28, 1])
		test_labels = mnist.test.labels
		self.mnist = mnist
	def get_state(self):
		train_inputs, train_labels = self.mnist.train.next_batch(1)
		return np.reshape(train_inputs, [-1,28,28,1])
def vis_loss(agent,test_inputs,test_labels):
	loss=[]
	for i in range(100):
		loss.append(agent.show_loss(test_inputs[[i]], test_labels[[i]]))
	print('Avg loss: {}'.format(np.mean(loss)))
	loss2 = agent.show_loss(test_inputs[:100], test_labels[:100])
	print('Avg loss by direct calculation: {}'.format(loss2))
	import matplotlib.pyplot as plt
	plt.plot(loss)
	plt.show()
def test_imaging():
	agent = optical_cnn(input_size=[28,28], output_size = 10, learning_rate=0.001)
	#'''
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	test_inputs = mnist.test.images
	test_inputs = np.reshape(test_inputs, [-1, 28, 28, 1])
	test_labels = mnist.test.labels
	from utils import visualize_modulator, visualize_sensor
	import os
	os.makedirs('./log', exist_ok=True)
	env = dummy_env_mnist()
	visualize_modulator(agent, './log/test_modulator.png')
	visualize_sensor(env, agent, './log/test_sensor.png')
	#'''

	#agent.save('./models/cnn')
	#img = agent.show_sensor(test_inputs[[0]])
	#img = img[0]
	print(agent.get_mod_params())
	#set_trace()

	# Visualize the loss
	agent.restore('./models/cnn')
	accuracy = agent.test(test_inputs, test_labels)
	print(accuracy)
	#vis_loss(agent,test_inputs,test_labels)
	pred = agent.sess.run(agent.pred, feed_dict={agent.input:test_inputs[[78]]})
	print(pred, test_labels[78])

	img=agent.show_sensor(test_inputs[[78]])
	import matplotlib.pyplot as plt
	plt.imshow(img[0,...,0], cmap='gray')
	#plt.imshow(test_inputs[78,...,0])
	plt.show()


if __name__ == "__main__":
	test_imaging()