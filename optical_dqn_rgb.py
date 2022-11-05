# import numpy as np
import tensorflow as tf
import numpy as np
import math
import pdb
import propagator_tool4 as ppg
from DQN_visrl_patchinput import *
tf.reset_default_graph()
class optical_dqn_rgb(DeepQNetwork):
	def __init__(self,unitchange=0.66e-4, fix_mod=False, fix_nn=False, learning_rate_mod=None, mod_resolution=40, cnn_activation='relu', *args, **kwargs):
		self.unitchange = unitchange
		self._fix_mod = fix_mod # If True, will not train the modulator.
		self._fix_nn  = fix_nn
		self.mod_resolution = mod_resolution # int. the resolution of the modulator = mod_resolution * mod_resolution
		activation_func = {
			'relu': tf.nn.relu
		}
		self.cnn_activation = activation_func[cnn_activation]
		super().__init__(*args, **kwargs)
		if learning_rate_mod is  None:
			learning_rate_mod = self.get_lr()
			self.lr_mod = tf.Variable(learning_rate_mod, trainable=False)
		elif len(learning_rate_mod) == 1:
			print('constant mod lr')
			learning_rate_mod=learning_rate_mod[0]
			self.lr_mod = tf.Variable(learning_rate_mod, trainable=False)
		else:
			lr_mod, decay_steps, decay_rate = learning_rate_mod
			self.lr_mod = tf.train.exponential_decay(lr_mod, self.global_step, decay_steps=int(decay_steps), decay_rate=decay_rate, staircase=False)
		
		# Separately set learning rate for modulator and Q-network parameters
		with tf.variable_scope('train'):
			increment_global_step_op = tf.assign(self.global_step, self.global_step+1)
			if self._fix_mod:
				opt2 = tf.train.AdamOptimizer(self.lr, name="Optim_rl")
				grads2 = tf.gradients(self.loss, self.rl_params)
				train_op2 = opt2.apply_gradients(zip(grads2, self.rl_params))
				self._train_op = tf.group(train_op2, increment_global_step_op)
			elif self._fix_nn:
				opt1 = tf.train.AdamOptimizer(self.lr_mod, name='Optim_mod')
				grads1 = tf.gradients(self.loss, self.mod_params)
				train_op1 = opt1.apply_gradients(zip(grads1, self.mod_params))
				self._train_op = tf.group(train_op1, increment_global_step_op)
			else:
				opt1 = tf.train.AdamOptimizer(self.lr_mod, name='Optim_mod')
				#opt1 = tf.train.RMSPropOptimizer(self.lr_mod, name='Optim_mod')
				opt2 = tf.train.AdamOptimizer(self.lr, name="Optim_rl")
				grads = tf.gradients(self.loss, self.mod_params + self.rl_params)
				grads1 = grads[:len(self.mod_params)]
				grads2 = grads[len(self.mod_params):]
				train_op1 = opt1.apply_gradients(zip(grads1, self.mod_params))
				train_op2 = opt2.apply_gradients(zip(grads2, self.rl_params))
				self._train_op = tf.group(train_op1, train_op2, increment_global_step_op)
		self.sess.run(tf.global_variables_initializer())
	def get_lr_mod(self):
		return self.sess.run(self.lr_mod)
	def set_lr_mod(self, learning_rate):
		self.sess.run(self.lr_mod.assign(learning_rate))
	def build_cnn(self, name, inputs):
		'''
		# single channel inputs: inputs shape is [N, H, W, C=1]
		with tf.variable_scope(name):
			img = self.optical_layer(inputs)
		'''
		# multi-channel inputs: inputs shape is [N, H, W, C]
		imgs = self.prop_multichannel(name, inputs)
		if name == 'eval_net_params': 
			self.imgs = imgs
			self.mod_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/modulator')
		if self._fix_mod: imgs = tf.stop_gradient(imgs)
		shape = imgs.shape
		imgs = tf.reshape(imgs, [-1, shape[1] * shape[2], shape[3]])
		imgs = tf.nn.l2_normalize(imgs, axis = 1)
		imgs = tf.reshape(imgs, [-1, shape[1], shape[2], shape[3]])
		imgs = imgs * 60
		return super().build_cnn(name,imgs, activation=self.cnn_activation)
	def prop_multichannel(self, name, inputs):
		"""
		Simulate optical wave propagation. The input is a batch of 3-color-channel images.
		inputs -- dimension is [n_batch, height, width, n_channel] with n_channel=3. The 3 channels correspond to R, G, B
		"""
		# Note: the input and output of optical_layer() is assumed to have shape: [None, dim, dim]
		# Adapt optical_layer to process multi-channel inputs
		wavelengths = {'R': 640e-9, 'G': 509e-9, 'B': 486e-9} # or 640, 509, 486 from https://hypertextbook.com/facts/2005/JustinChe.shtml
		refractive_indices = {'R':1.50917/1.51, 'G':1.51534/1.51, 'B':1.51690/1.51} # assume material is Borosilicate glass. data from https://hypertextbook.com/facts/2005/JustinChe.shtml
		shape = inputs.shape # [N, H, W, C]
		with tf.variable_scope(name+'/modulator'):
			imgs = self.optical_layer_rgb(inputs, wavelengths=wavelengths, refractive_indices=refractive_indices, dim_mod=self.mod_resolution, dim_hole=self.mod_resolution//2)
		#imgs = tf.reshape(imgs, (-1, shape[1], shape[2], shape[3]))
		#imgs = tf.transpose(imgs, [0,2,3,1])
		# debug: print the shape of original object and shape of sensor images
		print("shape of object: {} \nshape of sensor images: {}".format(shape, imgs.shape))
		
		#imgs = imgs / tf.reduce_max(imgs)
		return imgs
	def set_perfect_lens(self):
		self.sess.run(tf.assign(self.tf_mod_params, self._perfect_lens_param))
	def update_mask(self, ratio=1):
		dim_mod = self.dim_mod  # resolution of modulator
		dim_hole = int(dim_mod *ratio / np.sqrt(2))
		dim_mod_before_interp = self.dim_mod_before_interp  # resolution of interpolation points on modulator
		mask = ppg.mask_modulator_round(dim_mod, dim_hole)
		self.sess.run(tf.assign(self.mask, mask))
		dim_hole_interp = np.floor(dim_hole / dim_mod * dim_mod_before_interp).astype('int32')
		mask_interp = ppg.mask_modulator_round(dim_mod_before_interp, dim_hole_interp)
		#set_trace()
		self.sess.run(tf.assign(self.mask_interp, mask_interp))
	def show_modulator(self):
		modulator = self.sess.run(self.modulator_draw)
		modulator = modulator /1e-6#* self.unitchange / 1e-6
		return modulator
	def show_modulator_params(self):
		modulator = self.sess.run(self.tf_mod_params)
		modulator = modulator #* self.unitchange / 1e-6
		return modulator
	def reset_modulator(self):
		params = np.zeros([1,100,1]).astype('float32')
		self.sess.run(tf.assign(self.tf_mod_params, params))
		print('Modulator reset to flat surface')
	def load_modulator_from_npy(self, path):
		params = np.load(path)
		self.sess.run(tf.assign(self.tf_mod_params, params))
		print("Modulator loaded from {}".format(path))
	def show_sensor(self, inputs):
		imgs = self.sess.run(self.imgs, feed_dict={self.s:inputs})
		imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
		imgs = np.squeeze(imgs)
		return imgs
	def tf_interp(self):
		# params
		dim_mod_before_interp = self.dim_mod_before_interp
		dim_mod = self.dim_mod
		focal_length = self.focal_length
		len_modulator = self.len_modulator
		dim_mod_mask_interp = dim_mod_before_interp
		dim_hole = int(dim_mod / np.sqrt(2))
		dim_mod_mask = dim_mod
		dim_hole_interp = np.floor(dim_hole / dim_mod_mask * dim_mod_mask_interp).astype('int32')
		# add mask
		mask_interp = ppg.mask_modulator_round(dim_mod_mask_interp, dim_hole_interp)
		self.mask_interp = tf.Variable(mask_interp, trainable=False) # Set mask_interp as TF variable to change the aperture size
		modulator_before_interp_mask = self.tf_mod_params * tf.reshape(self.mask_interp, [1, dim_mod_mask_interp * dim_mod_mask_interp, 1])
		# interpolation
		initial_mod_lens_interp, train_points, query_points = ppg.lens_initialize_interp(dim_mod_before_interp, dim_mod, focal_length, len_modulator)
		modulator_interp = tf.contrib.image.interpolate_spline(
			train_points = train_points,
			train_values = tf.cast(modulator_before_interp_mask, tf.float64),
			query_points = query_points,
			order = 2,
			regularization_weight=0.0,
			name = 'interpolate_spline'
		)
		modulator_interp = tf.cast(modulator_interp, tf.float32) * self.unitchange
		return modulator_interp

	def optical_layer_rgb(self, 
			inputs,
			#dim_mod = 80  # resolution of modulator
			wavelengths,  # wavelengths
			refractive_indices,
			dim_mod = 40,  # 40
			dim_hole = 20, #20  # initial resolution of hole of the mask
			dim_mod_before_interp = 10, # resolution of interpolation points on modulator
			#focal_length = 1.5e-2, # unit is meter
			#unitchange = 5.e-4,  # change the unit of computation for a better accuracy
			):
		"""
		inputs:
			inputs -- input RGB image. shape is [N,H,W,C=3]
			wavelengths -- dict{'R':float, 'G':float, 'B':float}. wavelengths for R/G/B wavelength
		set params
		calculate the parameters for a perfect lens
		non-trainable tensors: propagators P1, P2 -- reuse for Q and Q'
		trainable params: modulator. M -- reuse for different RGB channels. refractive index???????????????
		add noise, normalize, reshape
		"""

		# Set parameters
		unitchange = self.unitchange # scale factor for SLM parameters
		dim_obj = int(inputs.shape[1]) # input shape is [N, H, W, C=3] with H = W
		dim_det = dim_obj
		dim_mod_mask = dim_mod
		self.dim_mod = dim_mod # used in interpolation functions
		self.dim_mod_before_interp = dim_mod_before_interp # used in interpolation functions
		dim_obj_total = dim_obj * dim_obj
		dim_det_total = dim_det * dim_det
		dim_mod_total = dim_mod * dim_mod
		# physical params
		scale = 1e-3
		len_modulator = .5 * scale # 0.5 * scale # length of modulator, unit is meter
		len_obj = 1 * scale#1e-3,
		dist_obj = 30 * scale
		dist_sensor = 30 * scale # originally 30 * scale
		len_det = dist_sensor * len_obj / dist_obj
		# params used in other functions
		self.len_modulator = len_modulator # used for interpolation
		self.focal_length = dist_obj * dist_sensor / (dist_obj + dist_sensor) # used in interpolation
		self.mask = ppg.mask_modulator_round(dim_mod_mask, dim_hole)
		#self.mask = tf.Variable(ppg.mask_modulator_round(dim_mod_mask, dim_hole), trainable = False) # if want to adjust mask size in training, set it as a TF variable
		# The perfect_lens_param is used for debug: check imaging of a perfect lens
		mod = ppg.lens_initialize(dim_mod_before_interp, self.focal_length, len_modulator) 
		base = ppg.lens_initialize(dim_mod, self.focal_length, len_modulator) 
		base = np.reshape(base, (dim_mod, dim_mod))
		base = base[dim_mod//2, dim_mod//2 - int(dim_hole*0.707)]
		mod = (mod - base) / (1.51) # the 1.51 is the refractive index
		self._perfect_lens_param = np.reshape(mod, [1, dim_mod_before_interp * dim_mod_before_interp, 1]).astype('float32')/ self.unitchange

		#############################################################################
		# set propagator matrices if it does not exist. Otherwise, read from dictionary. We use the same matrix for both Q and Q' computations to save GPU memory
		if not hasattr(self, 'propagators'):
			self.propagators = {} # propagators[R/G/B][1/2] = (real, imag)
			# iterate for RGB channels.
			for c in ['R', 'G', 'B']:
				propagator1_real, propagator1_imag = ppg.propagator_compute(dim_obj, len_obj, dim_mod, len_modulator, dist_obj,    wavelengths[c])
				propagator2_real, propagator2_imag = ppg.propagator_compute(dim_mod, len_modulator, dim_det, len_det, dist_sensor, wavelengths[c])
				self.propagators[c] = {
					1: (propagator1_real, propagator1_imag),
					2: (propagator2_real, propagator2_imag)
				}
		#############################################################################
		# Variable: thickness of 10*10 points on modulator
		initial_mod_unitchanged = np.zeros([1, dim_mod_before_interp * dim_mod_before_interp, 1]).astype('float32')
		self.tf_mod_params = tf.Variable(initial_mod_unitchanged, trainable = True, dtype = tf.float32, name = 'modulator_param') # this param means the 
		modulator_interp = self.tf_interp() 
		self.modulator = tf.reshape(modulator_interp, [dim_mod_total, 1])
		self.modulator_draw = tf.reshape(self.modulator, [dim_mod, dim_mod]) * tf.reshape(self.mask, [dim_mod, dim_mod]) # for modulator visualization
		# For debug: set modulator as a lens
		#self.modulator = tf.cast(tf.reshape(ppg.lens_initialize(dim_mod, focal_length, len_modulator), [dim_mod_total, 1]), tf.float32) / self.unitchange
		
		# propagate separately for each channel, and stack
		detector = {}
		channel_idx = {'R':0, 'G':1, 'B':2}
		for c in ['R', 'G', 'B']:
			detector[c] = self.propagate(
				inputs=inputs[:,:,:,channel_idx[c]],
				channel=c,
				mylambda=wavelengths[c],
				refractive_idx=refractive_indices[c],
				dim_obj=dim_obj,
				dim_mod=dim_mod,
				dim_det=dim_det,
				propagator1=self.propagators[c][1],
				propagator2=self.propagators[c][2]
			) # output shape [N, H, W]
		detector_rgb = tf.stack([detector['R'], detector['G'], detector['B']], axis=3) # shape [N, H, W, C=3]
		return detector_rgb
	def propagate(self, inputs, channel, mylambda, refractive_idx, dim_obj, dim_mod, dim_det, propagator1, propagator2):
		# Calculation of OutputIntensity = P2 * M * P1 * InputIntensity
		'''
		inputs:
			inputs -- input image. shape [N, H, W]
		returns:
			detector -- normalized sensor image. shape [N, H, W]
		1. params: dim_obj, dim_mod, dim_det, len_obj, len_modulator, len_det, dist_obj, dist_sensor, mylambda; (for mask) dim_mod_mask, dim_hole;
		2. input image: inputs; 
		3. 
		'''
		# InputIntensity, default phase is 0 on the whole plane
		'''
		# if set random phase on the object plane
		input_images = tf.reshape(tf.sqrt(inputs), [-1, dim_obj * dim_obj])      # calculate amplitude 
		object_amplitude = tf.transpose(input_images)    # change shape from [batch size, 28 * 28] to [28 * 28, batch size]
		object_random_phase = tf.zeros(tf.shape(object_amplitude)) * 2. * math.pi    # without random phase on input image
		object_real = object_amplitude * tf.cos(object_random_phase)
		object_imag = object_amplitude * tf.sin(object_random_phase)
		object_intensity = (object_real**2. + object_imag**2.) / (mylambda)**2. / (dim_obj * dim_mod)**4.
		'''
		inputs = tf.transpose(tf.reshape(inputs, [-1, dim_obj * dim_obj])) # change shape from [N, H, W] to [H * W, N]
		object_intensity = inputs / (mylambda)**2. / (dim_obj * dim_mod)**4. # scale the intensity
		# M, with mask on it to modulate amplitude
		# phase = thickness of modulator * 2pi / wavelength
		modulator_phase = self.modulator * 2. * math.pi / mylambda * refractive_idx # unit of length is 10^-6 m = 1 micron  
		phase_modulation_real = tf.reshape(tf.cos(modulator_phase),[-1, 1]) * tf.reshape(self.mask, [dim_mod * dim_mod, 1])
		phase_modulation_imag = tf.reshape(tf.sin(modulator_phase),[-1, 1]) * tf.reshape(self.mask, [dim_mod * dim_mod, 1])
		# extract real and imag parts of propagators
		propagator1_real, propagator1_imag = propagator1
		propagator2_real, propagator2_imag = propagator2
		# M * P1
		MP1_real = phase_modulation_real * propagator1_real - phase_modulation_imag * propagator1_imag
		MP1_imag = phase_modulation_imag * propagator1_real + phase_modulation_real * propagator1_imag
		# P2 * M * P1
		P2MP1_real = tf.matmul(propagator2_real, MP1_real) - tf.matmul(propagator2_imag, MP1_imag)
		P2MP1_imag = tf.matmul(propagator2_imag, MP1_real) + tf.matmul(propagator2_real, MP1_imag)
		P2MP1 = P2MP1_real**2. + P2MP1_imag**2.
		# OutputIntensity = |P2 * M * P1| * InputIntensity, for incoherent light
		detector = tf.matmul(P2MP1, object_intensity) # shape = [H*W, N]
		# add noise
		detector = self.detector_noise(detector)
		
		# the output is transformed into shape of [N, H, W]
		detector = tf.transpose(detector)
		detector = tf.reshape(detector, [-1, dim_det, dim_det])
		return detector
	def detector_noise(self, detector, noise_rate=0, propotional_random_rate=0):
		'''
		Add propotional noise and white noise on the detector
		detector_with_noise = tf.maximum(detector * (1.0 + propotional_random_noise) + white_noise, 0.)
		inputs: 
			detector -- detector image without noise
			noise_rate -- white noise level
			propotional_random_rate -- propotional noise level

		'''
		if propotional_random_rate > 0:
			# propotional noise 
			tfrandom_noise_propotional = tf.random_normal(
				shape = tf.shape(detector),
				mean=0.0,
				stddev=1.,
				dtype=tf.float32)
			propotional_random_noise = propotional_random_rate * tfrandom_noise_propotional     # with random noise on input image
			detector *= 1.0 + propotional_random_noise
		if noise_rate > 0:
			# white noise 
			tfrandom_noise = tf.random_normal(
				shape = tf.shape(detector), 
				mean=0.0,
				stddev=1.0,
				dtype=tf.float32)
			detector += tf.abs(tfrandom_noise) * noise_rate * tf.reduce_mean(detector) # add white noise
		return detector
	# Other utility functions
	def show_perfect_img(self, inputs):
		# This function does not work as expected. Probably the value of perfect_lens_param is wrong
		mod = self.show_modulator_params()
		#set_trace()
		self.sess.run(self.tf_mod_params.assign(self._perfect_lens_param))
		sensor = self.show_sensor(inputs)
		self.sess.run(self.tf_mod_params.assign(mod))
		return sensor

def test():
	# load image
	from PIL import Image
	import matplotlib.pyplot as plt
	objects = Image.open('demo.png').convert("RGB").resize([64,64])
	objects = np.array(objects).reshape([-1,64,64,3]).astype(np.float32)
	objects = objects / np.max(objects)
	#print(objects.shape, objects.dtype)
	#plt.imshow(objects[0])
	#plt.show()

	# build agent
	agent = optical_dqn_rgb(
            n_actions=5,
            n_input=(64,64,3))
	imgs = agent.show_perfect_img(objects)
	print(imgs.shape)
	plt.imshow(imgs)
	plt.show()
	import pdb; pdb.set_trace()
if __name__ == "__main__":
	test()