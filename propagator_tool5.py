##############################################
# updated from propagator_tool3.py
# edition3, started in Nov. 17, 2018 -- the distances are changed to
# absolote values instead of using wavelength as unit
# edition4, started in Nov. 17, 2018 -- 
# change from comoplex computation to real computation for tensorflow

# edition5, 5.17.2023
# use psf method to get rid of memory limit of full matrix calculation
##############################################
def propagator_compute(dim_obj, len_obj, dim_mod, len_mod, dist_obj2mod, mylambda):
    import tensorflow as tf
    import numpy as np
    import math
    import cmath
    import pdb
    randomloop = np.array(100);
    type_object = 'point'
    #     pdb.set_trace()
    type_modulator = 'lens'
    mylambda_unit = 10. ** 3.  # unit is 10**-3m
    mylambda = mylambda * mylambda_unit
    #     mylambda=tf.cast(5.*10.**-4., tf.float32) # unit is 10**-7m = 0.1 micron
    #     mylambda=np.float32(5.*10.**-7*10.**7.) # unit is 10**-7m = 0.1 micron
    pix_size_mod = np.array(dim_mod)
    #     len_object=np.array(2.*10.**3.*5.*10**-7)
    #     len_modulator=np.array(2.*10.**3.*5.*10**-7)
    #     focus_lens=np.array(3.*10.**4.*5.*10**-7)
    #     len_object=np.array(2.*10.**-3.*mylambda_unit)   # unit is 10**-7m = 0.1 micron
    #     len_modulator=np.array(2.*10.**-3.*mylambda_unit)  # unit is 10**-7m = 0.1 micron
    #     focus_lens=np.array(1.5*10.**-1.*mylambda_unit)   # unit is 10**-7m = 0.1 micron
    len_object = np.array(1. * len_obj * mylambda_unit)  # unit is 10**-7m = 0.1 micron
    len_modulator = np.array(1. * len_mod * mylambda_unit)  # unit is 10**-7m = 0.1 micron
    dist_obj2mod = np.array(1. * dist_obj2mod * mylambda_unit)  # unit is 10**-7m = 0.1 micron
    size_object = np.hstack([len_object, len_object])
    size_modulator = np.hstack([len_modulator, len_modulator])
    pix_size_obj = np.array([dim_obj, dim_obj])
    res_object = np.hstack([pix_size_obj[0], pix_size_obj[1]])
    res_modulator = np.hstack([pix_size_mod, pix_size_mod])

    coordx_o_linspace = ((np.arange(1., res_object[0] + 1.)) - (1. + res_object[0]) / 2.) / res_object[0] * size_object[
        0]
    coordy_o_linspace = ((np.arange(1., res_object[1] + 1.)) - (1. + res_object[1]) / 2.) / res_object[1] * size_object[
        1]

    coordx_m_linspace = ((np.arange(1., res_modulator[0] + 1.)) - (1. + res_modulator[0]) / 2.) / res_modulator[0] * \
                        size_modulator[0]
    coordy_m_linspace = ((np.arange(1., res_modulator[1] + 1.)) - (1. + res_modulator[1]) / 2.) / res_modulator[1] * \
                        size_modulator[1]

    coordx_o, coordy_o = np.meshgrid(coordx_o_linspace, coordy_o_linspace)
    coordx_o = coordx_o.T
    coordy_o = coordy_o.T
    coordx_m, coordy_m = np.meshgrid(coordx_m_linspace, coordy_m_linspace)
    coordx_m = coordx_m.T
    coordy_m = coordy_m.T

    coordx_o_kron = np.kron(np.ones(res_modulator.astype('int').tolist()), coordx_o)
    coordy_o_kron = np.kron(np.ones(res_modulator.astype('int').tolist()), coordy_o)
    coordx_m_kron = np.kron(coordx_m, np.ones(res_object.astype('int').tolist()))
    coordy_m_kron = np.kron(coordy_m, np.ones(res_object.astype('int').tolist()))
    #     pdb.set_trace()
    R_o2m = np.sqrt((coordx_o_kron - coordx_m_kron) ** 2. + (coordy_o_kron - coordy_m_kron) ** 2. + dist_obj2mod ** 2.)

    # for test
    deltaS_m = 1
    deltaS_o = 1
    deltaS_d = 1

    theta_o2m = np.arctan(
        np.sqrt((coordx_o_kron - coordx_m_kron) ** 2. + (coordy_o_kron - coordy_m_kron) ** 2.) / dist_obj2mod)
    #     propagator1_kron=np.exp(1j*2.*math.pi/mylambda*R_o2m)*deltaS_o*deltaS_m*np.cos(theta_o2m)/2.
    #     pdb.set_trace()
    propagator1_kron_real = tf.cos(2. * math.pi / mylambda * R_o2m) * deltaS_o * deltaS_m * np.cos(theta_o2m) / 2.
    propagator1_kron_imag = tf.sin(2. * math.pi / mylambda * R_o2m) * deltaS_o * deltaS_m * np.cos(theta_o2m) / 2.

    dimx1 = res_object[0]
    dimy1 = res_object[1]
    dimx2 = res_modulator[0]
    dimy2 = res_modulator[1]

    propagator1_4d_yyxx_real = tf.reshape(propagator1_kron_real, [dimx2, dimx1, dimy2, dimy1])
    propagator1_4d_yxyx_real = tf.transpose(propagator1_4d_yyxx_real, (0, 2, 1, 3))
    propagator1_2d_real = tf.reshape(propagator1_4d_yxyx_real, [dimy2 * dimx2, dimy1 * dimx1])

    propagator1_4d_yyxx_imag = tf.reshape(propagator1_kron_imag, [dimx2, dimx1, dimy2, dimy1])
    propagator1_4d_yxyx_imag = tf.transpose(propagator1_4d_yyxx_imag, (0, 2, 1, 3))
    propagator1_2d_imag = tf.reshape(propagator1_4d_yxyx_imag, [dimy2 * dimx2, dimy1 * dimx1])

    #     return tf.cast(propagator1_2d_real, tf.float32), tf.cast(propagator1_2d_imag, tf.float32)
    return propagator1_2d_real, propagator1_2d_imag


def lens_initialize(dim_mod, focal_lens, len_modulator):
    import tensorflow as tf
    import numpy as np
    import math
    import cmath
    import pdb
    # make sure there are points at the edge
    coordx = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator * dim_mod / (dim_mod - 1);
    coordy = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator * dim_mod / (dim_mod - 1);
    # coordx = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator;
    # coordy = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator;
    coordx_matrix, coordy_matrix = np.meshgrid(coordx, coordy);
    coordx_matrix = coordx_matrix.T
    coordy_matrix = coordy_matrix.T
    coordr_matrix = np.sqrt(coordx_matrix ** 2. + coordy_matrix ** 2.)
    #     amplitude_matrix = coordr_matrix<len_modulator/2;
    distance_phase_matrix = ((len_modulator * np.sqrt(2.) / 2.) ** 2. - coordx_matrix ** 2. - coordy_matrix ** 2.) / 2. / focal_lens;
    
    distance_phase_matrix = np.reshape(distance_phase_matrix, [dim_mod ** 2, 1])
    # print(coordx/len_modulator)
    # pdb.set_trace()
    return distance_phase_matrix


# for interpolation, the boundary points are exact on the boundary line of the modulator
def lens_initialize_interp(dim_mod, dim_query, focal_lens, len_modulator):
    import tensorflow as tf
    import numpy as np
    import math
    import cmath
    import pdb
    coordx = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator * dim_mod / (dim_mod - 1);
    coordy = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator * dim_mod / (dim_mod - 1);
    coordx_matrix, coordy_matrix = np.meshgrid(coordx, coordy);
    coordx_matrix = coordx_matrix.T
    coordy_matrix = coordy_matrix.T
    train_points = np.zeros([dim_mod, dim_mod, 2])
    train_points[:, :, 0] = coordx_matrix
    train_points[:, :, 1] = coordy_matrix
    train_points1 = np.reshape(train_points, [1, dim_mod * dim_mod, 2])

    coordx_query = (np.arange(1, dim_query + 1) - (1 + dim_query) / 2.) / dim_query * len_modulator
    coordy_query = (np.arange(1, dim_query + 1) - (1 + dim_query) / 2.) / dim_query * len_modulator
    coordx_query_matrix, coordy_query_matrix = np.meshgrid(coordx_query, coordy_query);
    coordx_query_matrix = coordx_query_matrix.T
    coordy_query_matrix = coordy_query_matrix.T
    query_points = np.zeros([dim_query, dim_query, 2])
    query_points[:, :, 0] = coordx_query_matrix
    query_points[:, :, 1] = coordy_query_matrix
    query_points1 = np.reshape(query_points, [1, dim_query * dim_query, 2])

    coordr_matrix = np.sqrt(coordx_matrix ** 2. + coordy_matrix ** 2.)
    #     amplitude_matrix = coordr_matrix<len_modulator/2;
    distance_phase_matrix = ((len_modulator * np.sqrt(2.) / 2.) ** 2. - coordx_matrix ** 2. - coordy_matrix ** 2.) / 2. / focal_lens;
    distance_phase_matrix = np.reshape(distance_phase_matrix, [1, dim_mod ** 2, 1])
    #     pdb.set_trace()
    return distance_phase_matrix, train_points1, query_points1


# in order to add a mask on the modulator during the training, which means zero on the mask
# square mask
def mask_modulator(dim_mod, dim_hole):
    import numpy as np
    if dim_hole >= dim_mod:
        dim_hole = dim_mod
    index_start = np.ceil((dim_mod - dim_hole) / 2.)
    index_end = np.floor((dim_mod + dim_hole) / 2.)
    mask = np.zeros([dim_mod, dim_mod])
    index = np.arange(index_start, index_end)
    mask[np.ix_(index, index)] = 1.
    return mask


# round mask
def mask_modulator_round(dim_mod, dim_hole):
    import numpy as np
    coord_x = np.arange(0, dim_mod) - (dim_mod - 1.) / 2.
    coord_y = np.arange(0, dim_mod) - (dim_mod - 1.) / 2.
    coord_x_matrix, coord_y_matrix = np.meshgrid(coord_x, coord_y)
    coord_x_matrix = coord_x_matrix.T
    coord_y_matrix = coord_y_matrix.T
    coord_r_matrix = np.sqrt(coord_x_matrix ** 2. + coord_y_matrix ** 2.)
    index_matrix = coord_r_matrix < dim_hole / 2. * np.sqrt(2.)
    mask = np.zeros([dim_mod, dim_mod], dtype=np.float32)
    mask[index_matrix] = 1.
    return mask


def data_generator(batch_size, data, data_label):
    import numpy as np
    while (True):
        indexes = np.arange(0, data.shape[0], 1)
        np.random.shuffle(indexes)
        max_range = int(data.shape[0] / batch_size)
        for i in range(max_range):
            data_temp = np.array([data[k] for k in indexes[i * batch_size:(i + 1) * batch_size]])
            data_label_temp = np.array([data_label[k] for k in indexes[i * batch_size:(i + 1) * batch_size]])
            yield data_temp, data_label_temp


def initial_object(pix_modulator, type_object, coordxy_normalized):
    import numpy as np
    if type_object == 'point':
        amplitude_object = np.zeros(pix_modulator).astype(np.float32)
        amplitude_object[coordxy_normalized[0], coordxy_normalized[1]] = 1.
    else:
        raise ValueError("other object type hasn't been developed yet")
    return amplitude_object.astype(np.float32)


def psf_func_cal(res_object, dist_obj2mod, size_object, res_modulator, lambda_):
    import numpy as np
    coordx_o_linspace = (np.arange(1, res_object[0] + 1) - (1 + res_object[0]) / 2) / res_object[0] * size_object[0]
    
    coordy_o_linspace = (np.arange(1, res_object[1] + 1) - (1 + res_object[1]) / 2) / res_object[1] * size_object[1]
    deltax_length = coordx_o_linspace[1] - coordx_o_linspace[0]
    deltay_length = coordy_o_linspace[1] - coordy_o_linspace[0]

    res_kernel = res_modulator + res_object - 1
    coordx_kernel_linspace = ((np.arange(1, res_kernel[0] + 1) - (1 + res_kernel[0]) / 2)) * deltax_length
    coordy_kernel_linspace = ((np.arange(1, res_kernel[1] + 1) - (1 + res_kernel[1]) / 2)) * deltay_length

    coordx_kernel, coordy_kernel = np.meshgrid(coordx_kernel_linspace, coordy_kernel_linspace)
    coordx_kernel = coordx_kernel.T
    coordy_kernel = coordy_kernel.T

    deltaS_o = (coordx_o_linspace[1] - coordx_o_linspace[0]) * (coordy_o_linspace[1] - coordy_o_linspace[0])
    theta_o2m = np.arctan(np.sqrt(coordx_kernel ** 2 + coordy_kernel ** 2) / dist_obj2mod)
    R_o2m = np.sqrt(coordx_kernel ** 2 + coordy_kernel ** 2 + dist_obj2mod ** 2)
    propagator_kernel_real = np.cos(2. * np.pi / lambda_ * R_o2m) / R_o2m * deltaS_o * \
                             np.cos(theta_o2m) / (2. * lambda_)
    propagator_kernel_imag = np.sin(2. * np.pi / lambda_ * R_o2m) / R_o2m * deltaS_o * \
                             np.cos(theta_o2m) / (2. * lambda_)

    return propagator_kernel_real.astype(np.float32), propagator_kernel_imag.astype(np.float32)


def psf_func_cal_incoherence(res_object_new, dist_obj2mod, size_object, res_modulator,
                             lambda_, dist_mod2det, size_modulator, res_detector,
                             phase_modulation_real, phase_modulation_imag):
    # phase_modulation_real, phase_modulation_imag
    import numpy as np
    import tensorflow as tf
    coordxy = np.array([0, 0]).astype(np.float32) / res_object_new * size_object
    coordxy_normalized = np.round((coordxy + size_object / 2.) / size_object * res_object_new).astype(np.int32)

    amplitude_object_new = initial_object(res_object_new, 'point', coordxy_normalized)
    amplitude_object_new_real = amplitude_object_new
    amplitude_object_new_imag = np.zeros_like(amplitude_object_new_real).astype(np.float32)

    propagator1_kernel_real, propagator1_kernel_imag = psf_func_cal(res_object_new, dist_obj2mod, size_object,
                                                                    res_modulator, lambda_)

    res_detector_enlarge = res_detector + res_object_new - 1
    propagator2_kernel_real, propagator2_kernel_imag = psf_func_cal(res_modulator, dist_mod2det, size_modulator,
                                                                    res_detector_enlarge, lambda_)

    amplitude_front_mod_real = expand_and_conv2d(propagator1_kernel_real, amplitude_object_new_real) - \
                               expand_and_conv2d(propagator1_kernel_imag, amplitude_object_new_imag)
    amplitude_front_mod_imag = expand_and_conv2d(propagator1_kernel_real, amplitude_object_new_imag) + \
                               expand_and_conv2d(propagator1_kernel_imag, amplitude_object_new_real)

    amplitude_behind_mod_real = amplitude_front_mod_real * phase_modulation_real - \
                                amplitude_front_mod_imag * phase_modulation_imag

    amplitude_behind_mod_imag = amplitude_front_mod_real * phase_modulation_imag + \
                                amplitude_front_mod_imag * phase_modulation_real

    amplitude_detector_real = expand_and_conv2d(propagator2_kernel_real, amplitude_behind_mod_real) - \
                              expand_and_conv2d(propagator2_kernel_imag, amplitude_behind_mod_imag)
    amplitude_detector_imag = expand_and_conv2d(propagator2_kernel_real, amplitude_behind_mod_imag) + \
                              expand_and_conv2d(propagator2_kernel_imag, amplitude_behind_mod_real)

    propagator_kernel_incoherence = tf.square(amplitude_detector_real) + \
                                    tf.square(amplitude_detector_imag)

    return propagator_kernel_incoherence


def psf_func_cal_incoherence_optim(dim_mod, PSF1_real, PSF1_imag, PSF2_real, PSF2_imag,
                                   phase_modulation_real, phase_modulation_imag):
    import tensorflow as tf

    amplitude_behind_mod_real = PSF1_real * phase_modulation_real - \
                                PSF1_imag * phase_modulation_imag

    amplitude_behind_mod_imag = PSF1_real * phase_modulation_imag + \
                                PSF1_imag * phase_modulation_real

    amplitude_detector_real = expand_and_conv2d(PSF2_real, amplitude_behind_mod_real) - \
                              expand_and_conv2d(PSF2_imag, amplitude_behind_mod_imag)
    amplitude_detector_imag = expand_and_conv2d(PSF2_real, amplitude_behind_mod_imag) + \
                              expand_and_conv2d(PSF2_imag, amplitude_behind_mod_real)

    propagator_kernel_incoherence = tf.square(amplitude_detector_real) + \
                                    tf.square(amplitude_detector_imag)

    return propagator_kernel_incoherence


def expand_and_conv2d(propagator_kernel_incoherence_tensor, amplitude_object_new_tensor):
    import numpy as np
    import tensorflow as tf
    # Assuming propagator_kernel_incoherence and amplitude_object_new are 2-d numpy arrays
    # We convert these numpy arrays to tensors
    # propagator_kernel_incoherence_tensor = tf.convert_to_tensor(propagator_kernel_incoherence)
    # amplitude_object_new_tensor = tf.convert_to_tensor(amplitude_object_new)

    # Expand dimensions to make it a 4D tensor for tf.nn.convolution
    # This is necessary because TensorFlow's convolution functions operate on batches of 3D tensors
    propagator_kernel_incoherence_tensor_4d = tf.expand_dims(
        tf.expand_dims(propagator_kernel_incoherence_tensor, axis=-1), axis=0)
    amplitude_object_new_tensor_4d = tf.expand_dims(tf.expand_dims(amplitude_object_new_tensor, axis=-1), axis=-1)
    # Perform the 2D convolution

    intensity_detector = tf.nn.convolution(input=propagator_kernel_incoherence_tensor_4d,
                                           filter=amplitude_object_new_tensor_4d,
                                           padding='VALID')

    # Squeeze the output to make it a 2D tensor again
    intensity_detector = tf.squeeze(intensity_detector)
    return intensity_detector


def expand_and_conv2d_batch(propagator_kernel_incoherence_tensor, amplitude_object_new_tensor):
    import numpy as np
    import tensorflow as tf

    # with batch size > 1, shape of amplitude_object_new is [dim_mod,dim_mod,batch_size]
    # propagator_kernel_incoherence shape = [dim_ker, dim_ker], where dim_ker = dim_obj + dim_det - 1
    # input shape = [batch_size] + [dim1, dim1] + [in_channels = 1]
    # filter shape = [dim2,dim2] + [in_channels = 1, out_channels = 1]

    dim_obj_list = amplitude_object_new_tensor.get_shape().as_list()
    dim_obj = dim_obj_list[0]
    dim_ker_list = propagator_kernel_incoherence_tensor.get_shape().as_list()
    dim_ker = dim_ker_list[0]
    dim_det = dim_ker - dim_obj + 1

    # Assuming propagator_kernel_incoherence and amplitude_object_new are 2-d numpy arrays
    # We convert these numpy arrays to tensors
    # propagator_kernel_incoherence_tensor = tf.convert_to_tensor(propagator_kernel_incoherence)
    # amplitude_object_new_tensor = tf.convert_to_tensor(amplitude_object_new)

    # Expand dimensions to make it a 4D tensor for tf.nn.convolution
    # This is necessary because TensorFlow's convolution functions operate on batches of 3D tensors
    amplitude_object_new_tensor_transpose = tf.transpose(amplitude_object_new_tensor, perm=[2, 0, 1])

    amplitude_object_new_tensor_4d = tf.expand_dims(amplitude_object_new_tensor_transpose, axis=-1)

    propagator_kernel_incoherence_tensor_4d = tf.expand_dims(
        tf.expand_dims(propagator_kernel_incoherence_tensor, axis=-1), axis=-1)

    # Perform the 2D convolution
    # convolution function cannot be directly used due to fileter spacial size is larger than that of input
    # padding is a way to get rid of this problem

    pad_n = np.floor((dim_ker - dim_obj + 1) / 2).astype(
        np.int32)  # any number making input matrix larger than filter is fine

    padded_input = tf.pad(amplitude_object_new_tensor_4d, [[0, 0], [pad_n, pad_n], [pad_n, pad_n], [0, 0]], "CONSTANT")

    intensity_detector_4d = tf.nn.convolution(input=padded_input,
                                             filter=propagator_kernel_incoherence_tensor_4d,
                                             padding='SAME')

    crop_index = pad_n + (dim_obj - dim_det) / 2
    crop_index = crop_index.astype(np.int32)
    intensity_detector_4d = intensity_detector_4d[:, crop_index:-crop_index, crop_index:-crop_index, :]
    # output of convolution is [batch_size] + [dim3, dim3] + [out_channels = 1], we need output shape = [dim3*dim3] + [batch_size]
    # Squeeze the output to make it a 2D tensor again
    intensity_detector_3d = tf.squeeze(intensity_detector_4d)

    intensity_detector = tf.reshape(intensity_detector_3d, [-1, dim_det * dim_det])
    intensity_detector = tf.transpose(intensity_detector)
    return intensity_detector


def expand_and_conv2d_fft_batch(PSF, obj):
    import numpy as np
    import tensorflow as tf

    # with batch size > 1, shape of amplitude_object_new is [dim_mod,dim_mod,batch_size]
    # propagator_kernel_incoherence shape = [dim_ker, dim_ker], where dim_ker = dim_obj + dim_det - 1
    # input shape = [batch_size] + [dim1, dim1] + [in_channels = 1]
    # filter shape = [dim2,dim2] + [in_channels = 1, out_channels = 1]

    N = PSF.get_shape().as_list()[0]
    N_ker = obj.get_shape().as_list()[0]

    N_pad = int((N - N_ker)) // 2
    N_final = N - N_ker + 1
    N_ker_half = int((N_ker - 1) // 2)
    
    if (N - N_ker) % 2 == 0:
    	paddings = tf.constant([[0, 0], [N_pad, N_pad], [N_pad, N_pad], [0, 0]])
    else:
    	paddings = tf.constant([[0, 0], [N_pad, N_pad + 1], [N_pad, N_pad + 1], [0, 0]])

    obj_4d = tf.expand_dims(tf.transpose(obj, perm=[2, 0, 1]), axis=-1)  # ker

    PSF_4d = tf.expand_dims(tf.expand_dims(PSF, axis=-1), axis=-1)  # img

    ker_pad = tf.pad(obj_4d, paddings, 'CONSTANT')

    img_reshape = tf.transpose(PSF_4d, perm=[2, 3, 0, 1])
    ker_pad_reshape = tf.transpose(ker_pad, perm=[3, 0, 1, 2])

    img_fft = tf.fft2d(tf.cast(img_reshape, dtype=tf.complex64))
    ker_fft = tf.fft2d(tf.cast(ker_pad_reshape, dtype=tf.complex64))
    ker_fft = tf.conj(ker_fft)

    output_fft = tf.multiply(img_fft, ker_fft)

    # Inverse FFT
    output_fft_conv = tf.ifft2d(output_fft)
    output_fft_conv = tf.signal.fftshift(output_fft_conv, axes=[2, 3])
    output_fft_conv = tf.real(output_fft_conv)
    output_fft_conv = tf.transpose(output_fft_conv, perm=[1, 2, 3, 0])
    # Crop
    output_fft_conv = output_fft_conv[:, N_ker_half:N_ker_half + N_final, N_ker_half:N_ker_half + N_final, :]

    intensity_detector = tf.reshape(tf.squeeze(output_fft_conv), [-1, N_final * N_final])
    return tf.transpose(intensity_detector)


