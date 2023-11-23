# import numpy as np
import tensorflow as tf
import numpy as np
import math
import pdb
import propagator_tool5 as ppg
from DQN_visrl_patchinput import *
tf.reset_default_graph()
class optical_dqn_rgb(DeepQNetwork):
	def __init__(self,unitchange=1e-3,
				 fix_mod=False,
				 fix_nn=False,
				 learning_rate_mod=None,
				 mod_resolution=40,
				 cnn_activation='relu', *args, **kwargs):
		self.unitchange = unitchange
		self._fix_mod = fix_mod # If True, will not train the modulator.
		self._fix_nn  = fix_nn
		self.mod_resolution = mod_resolution # int. the resolution of the modulator = mod_resolution * mod_resolution
		activation_func = {
			'relu': tf.nn.relu,
			'elu': tf.nn.elu
		}
		self.cnn_activation = activation_func[cnn_activation]
		super().__init__(*args, **kwargs)
		#if self._fix_mod:
		#	self.set_perfect_lens() # update 2020.11.11: the fix_mod will not set modulator to a perfect lens.
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
		# Note 2021.3.16: we find that the normalization makes the color unreal.
		# but the normalization may be helpful to CNN training.
		shape = imgs.shape
		imgs = tf.reshape(imgs, [-1, shape[1] * shape[2], shape[3]])
		imgs = tf.nn.l2_normalize(imgs, axis = 1)
		imgs = tf.reshape(imgs, [-1, shape[1], shape[2], shape[3]])
		imgs = imgs * 60
		return super().build_cnn(name, imgs, activation=self.cnn_activation)

	def prop_multichannel(self, name, inputs):
		# Note: the input and output of optical_layer() is assumed to have shape: [None, dim, dim]
		# Adapt optical_layer to process multi-channel inputs
		'''
		update 2021.3.15: assign different wavelengths for R/G/B channels. Note that since we call the function optical_layer multiple times, we should reuse the TF params unrelated to the wavelength
		'''
		wavelengths = {'R': 640e-9, 'G': 509e-9, 'B': 486e-9} # or 640, 509, 486 from https://hypertextbook.com/facts/2005/JustinChe.shtml
		# wavelengths = {'R': 500e-9, 'G': 500e-9, 'B': 500e-9}
		refractive_indices = {'R':1.50917/1.51, 'G':1.51534/1.51, 'B':1.51690/1.51} # assume material is Borosilicate glass. data from https://hypertextbook.com/facts/2005/JustinChe.shtml
		#refractive_indices = {'R':1.51, 'G':1.51, 'B':1.51}
		shape = inputs.shape # [N, H, W, C]
		#imgs = tf.transpose(inputs, [0,3,1,2]) # [N, C, H, W]
		#imgs = tf.reshape(imgs, (-1, shape[1], shape[2])) # [N * C, H, W]
		with tf.variable_scope(name+'/modulator'):
			imgs = self.optical_layer_rgb(inputs, wavelengths=wavelengths, refractive_indices=refractive_indices, dim_mod=self.mod_resolution, dim_hole=self.mod_resolution//2)
		#imgs = tf.reshape(imgs, (-1, shape[1], shape[2], shape[3]))
		#imgs = tf.transpose(imgs, [0,2,3,1])
		# debug: print the shape of original object and shape of sensor images
		print("shape of object: {} \nshape of sensor images: {}".format(shape, imgs.shape))
		
		#imgs = imgs / tf.reduce_max(imgs)
		return imgs

	def optical_layer_rgb(self,
						  inputs,
						  wavelengths,  # wavelengths
						  refractive_indices,
						  dim_mod = 40,  # resolution of modulator
						  dim_hole = 20,  # initial resolution of hole of the mask
						  dim_mod_before_interp = 10, # resolution of interpolation points on modulator
						  # focal_length = 1.5e-2, # unit is meter
						  # unitchange = 5.e-4,  # change the unit of computation for a better accuracy
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
		# print('dim obj:', dim_obj)
		dim_det = dim_obj
		dim_mod_mask = dim_mod
		self.dim_mod = dim_mod # used in interpolation functions
		self.dim_mod_before_interp = dim_mod_before_interp # used in interpolation functions
		dim_obj_total = dim_obj * dim_obj
		dim_det_total = dim_det * dim_det
		dim_mod_total = dim_mod * dim_mod
		# physical params
		scale = 1e-3
		len_obj = 1.0 * scale
		len_modulator = len_obj * dim_mod / dim_obj  # length of modulator, unit is meter
		
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
		
		mod = mod - base  # mod is thickness, without unitchange!
		# We don't need to divide by 1.51, since that has been included in refractive index definition!
		self._perfect_lens_param = np.reshape(mod, [1, dim_mod_before_interp * dim_mod_before_interp, 1]).astype('float32') / self.unitchange
		
		# Construct the phase mask
		# Variable: thickness of 10*10 points on modulator
		initial_mod_unitchanged = np.zeros([1, dim_mod_before_interp * dim_mod_before_interp, 1]).astype('float32')
		
		self.tf_mod_params = tf.Variable(initial_mod_unitchanged, trainable = True, dtype = tf.float32, name = 'modulator_param')
		# self.tf_mod_params = tf.Variable(self._perfect_lens_param, trainable = True, dtype = tf.float32, name = 'modulator_param')  # For debug: set modulator as a lens
		# Note that "self.tf_mod_params" should be "unit changed"!
		
		modulator_interp = self.tf_interp()
		self.modulator = tf.reshape(modulator_interp, [dim_mod_total, 1])
		# Compare with perfect modulator (just for debug)
		# self.perfect_modulator = tf.cast(tf.reshape(ppg.lens_initialize(dim_mod, self.focal_length, len_modulator), [dim_mod_total, 1]), tf.float32)
		
		self.modulator_draw = tf.reshape(self.modulator, [dim_mod, dim_mod]) * tf.reshape(self.mask, [dim_mod, dim_mod]) # for modulator visualization
		# self.modulator: actual distance

		#############################################################################
		# set propagator matrices if it does not exist. Otherwise, read from dictionary. We use the same matrix for both Q and Q' computations to save GPU memory
		if not hasattr(self, 'propagators'):
			self.propagators = {} # propagators[R/G/B][1/2] = (real, imag)
			# iterate for RGB channels.
			for c in ['R', 'G', 'B']:
				'''
				# The original version
				propagator1_real, propagator1_imag = ppg.propagator_compute(dim_obj, len_obj, dim_mod, len_modulator, dist_obj,    wavelengths[c])
				propagator2_real, propagator2_imag = ppg.propagator_compute(dim_mod, len_modulator, dim_det, len_det, dist_sensor, wavelengths[c])
				'''
				# New version for calculating propagator
				size_object = np.array([len_obj, len_obj])
				size_modulator = np.array([len_modulator, len_modulator])

				res_object = np.array([dim_obj, dim_obj])
				res_modulator = np.array([dim_mod, dim_mod])
				res_detector = np.array([dim_det, dim_det])

				res_detector_enlarge = res_detector + res_object - 1
				
				propagator1_real, propagator1_imag = \
					ppg.psf_func_cal(res_object, dist_obj, size_object, res_modulator, wavelengths[c])

				# Crop the first PSF
				dim_PSF1 = np.shape(propagator1_real)[0]
				start_i = int((dim_PSF1 - dim_mod) // 2)
				end_i = start_i + dim_mod
				propagator1_real = propagator1_real[start_i:end_i, start_i:end_i]
				propagator1_imag = propagator1_imag[start_i:end_i, start_i:end_i]

				propagator2_real, propagator2_imag = \
					ppg.psf_func_cal(res_modulator, dist_sensor, size_modulator, res_detector_enlarge, wavelengths[c])
				# Save the results
				self.propagators[c] = {
					1: (propagator1_real, propagator1_imag),
					2: (propagator2_real, propagator2_imag)
				}

	#############################################################################
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
		
	def propagate(self, inputs, channel, mylambda, refractive_idx,
				  dim_obj, dim_mod, dim_det, propagator1, propagator2):
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
		# Step 1: prepare the input image
		inputs = tf.transpose(tf.reshape(inputs, [-1, dim_obj * dim_obj])) # change shape from [N, H, W] to [H * W, N]
		# object_intensity = inputs / (mylambda)**2. / (dim_obj * dim_mod)**4. # scale the intensity
		object_intensity = inputs * (mylambda)**2 / (500e-9 ** 4) / (dim_obj * dim_mod)**4
		object_intensity_3d = tf.reshape(object_intensity, [dim_obj, dim_obj, -1])

		# Step 2: prepare the phase modulator
		# phase = thickness of modulator * 2pi / wavelength
		modulator_phase = self.modulator * 2. * math.pi / mylambda * refractive_idx  # unit of length is 10^-6 m = 1 micron

		phase_modulation_real = tf.reshape(tf.cos(modulator_phase),[-1, 1]) * tf.reshape(self.mask, [dim_mod * dim_mod, 1])
		phase_modulation_imag = tf.reshape(tf.sin(modulator_phase),[-1, 1]) * tf.reshape(self.mask, [dim_mod * dim_mod, 1])
		# Reshape it into 2D
		phase_modulation_real_2d = tf.reshape(phase_modulation_real, [dim_mod, dim_mod])
		phase_modulation_imag_2d = tf.reshape(phase_modulation_imag, [dim_mod, dim_mod])

		# Step 3: prepare the PSF for incoherent light
		# extract real and imag parts of propagators
		(propagator1_real, propagator1_imag) = propagator1
		(propagator2_real, propagator2_imag) = propagator2
		
		# print('dim mod:', dim_mod)
		self.PSF_incoherence = ppg.psf_func_cal_incoherence_optim(
			dim_mod, propagator1_real, propagator1_imag, propagator2_real,
			propagator2_imag, phase_modulation_real_2d, phase_modulation_imag_2d)

		# Step 4: calculate the detector output
		'''
		# The original version: uses matrices, which is inefficient
		# M * P1
		MP1_real = phase_modulation_real * propagator1_real - phase_modulation_imag * propagator1_imag
		MP1_imag = phase_modulation_imag * propagator1_real + phase_modulation_real * propagator1_imag
		# P2 * M * P1
		P2MP1_real = tf.matmul(propagator2_real, MP1_real) - tf.matmul(propagator2_imag, MP1_imag)
		P2MP1_imag = tf.matmul(propagator2_imag, MP1_real) + tf.matmul(propagator2_real, MP1_imag)
		P2MP1 = P2MP1_real**2. + P2MP1_imag**2.
		# OutputIntensity = |P2 * M * P1| * InputIntensity, for incoherent light
		detector = tf.matmul(P2MP1, object_intensity) # shape = [H*W, N]
		'''
		# detector = ppg.expand_and_conv2d_batch(self.PSF_incoherence, object_intensity_3d)
		detector = ppg.expand_and_conv2d_fft_batch(self.PSF_incoherence, object_intensity_3d)

		# add noise
		detector = self.detector_noise(detector)

		# the output is transformed into shape of [N, H, W]
		detector = tf.transpose(detector)
		detector = tf.reshape(detector, [-1, dim_det, dim_det])
		return detector

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
		modulator = modulator / 1e-6
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
		
	def show_propagator(self, inputs):
		propagator = self.propagators# self.sess.run(self.propagators, feed_dict={self.s:inputs})
		propagator = propagator['G']
		propagator1 = propagator[1]
		propagator2 = propagator[2]
		return propagator1, propagator2
	def show_PSF(self, inputs):
		PSF = self.sess.run(self.PSF_incoherence, feed_dict={self.s:inputs})
		return PSF
		
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
		self.mask_interp = tf.Variable(mask_interp, trainable=False, dtype=tf.float32) # Set mask_interp as TF variable to change the aperture size
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

	def get_lr_mod(self):
		return self.sess.run(self.lr_mod)
	def set_lr_mod(self, learning_rate):
		self.sess.run(self.lr_mod.assign(learning_rate))

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
	img_size = 64  # Original: 64
	
	objects = Image.open('demo.png').convert("RGB").resize([img_size, img_size])
	objects = np.array(objects).reshape([-1,img_size,img_size,3]).astype(np.float32)
	
	# objects = np.zeros([1,img_size,img_size,3]).astype(np.float32)
	# objects[0, img_size // 2, img_size // 2, 1] = 1
	
	objects = objects / np.max(objects)
	#print(objects.shape, objects.dtype)
	
	plt.figure(0)
	plt.imshow(objects[0, :, :, :])
	#plt.show()

	# build agent
	agent = optical_dqn_rgb(
            n_actions=5,
            n_input=(img_size,img_size,3),
            mod_resolution=img_size // 1)
        
        # Obtain some variables, for debug
	imgs = agent.show_sensor(objects)
	
	modulator = agent.show_modulator()
	# propagator1, propagator2 = agent.show_propagator(objects)
	
	# (propagator1_real, propagator1_imag) = propagator1
	# (propagator2_real, propagator2_imag) = propagator2
	
	PSF = agent.show_PSF(objects)
	
	print('imgs shape:', np.shape(imgs))
	
	plt.figure(1)
	plt.imshow(modulator)
	plt.colorbar()
	
	plt.figure(2)
	plt.imshow(PSF)
	plt.colorbar()
	
	plt.figure(3)
	plt.imshow(imgs)
	plt.colorbar()
	
	plt.show()
	
	# import pdb; pdb.set_trace()

if __name__ == "__main__":
	test()

