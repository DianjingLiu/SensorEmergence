##############################################
# updated from propagator_tool3.py
# edition3, started in Nov. 17, 2018 -- the distances are changed to
# absolote values instead of using wavelength as unit
# edition4, started in Nov. 17, 2018 -- 
# change from comoplex computation to real computation for tensorflow
##############################################
def propagator_compute(dim_obj, len_obj, dim_mod, len_mod, dist_obj2mod, mylambda):
    import tensorflow as tf
    import numpy as np
    import math
    import cmath
    import pdb
    randomloop=np.array(100);
    type_object='point'
#     pdb.set_trace()
    type_modulator='lens'
    mylambda_unit=10.**3.  # unit is 10**-3m
    mylambda = mylambda * mylambda_unit
#     mylambda=tf.cast(5.*10.**-4., tf.float32) # unit is 10**-7m = 0.1 micron
#     mylambda=np.float32(5.*10.**-7*10.**7.) # unit is 10**-7m = 0.1 micron
    pix_size_mod=np.array(dim_mod)
#     len_object=np.array(2.*10.**3.*5.*10**-7)
#     len_modulator=np.array(2.*10.**3.*5.*10**-7)
#     focus_lens=np.array(3.*10.**4.*5.*10**-7)
#     len_object=np.array(2.*10.**-3.*mylambda_unit)   # unit is 10**-7m = 0.1 micron
#     len_modulator=np.array(2.*10.**-3.*mylambda_unit)  # unit is 10**-7m = 0.1 micron
#     focus_lens=np.array(1.5*10.**-1.*mylambda_unit)   # unit is 10**-7m = 0.1 micron
    len_object=np.array(1.*len_obj*mylambda_unit)   # unit is 10**-7m = 0.1 micron
    len_modulator=np.array(1.*len_mod*mylambda_unit)  # unit is 10**-7m = 0.1 micron
    dist_obj2mod=np.array(1.*dist_obj2mod*mylambda_unit)   # unit is 10**-7m = 0.1 micron
    size_object=np.hstack([len_object,len_object])
    size_modulator=np.hstack([len_modulator,len_modulator])
    pix_size_obj=np.array([dim_obj, dim_obj])
    res_object=np.hstack([pix_size_obj[0], pix_size_obj[1]])
    res_modulator=np.hstack([pix_size_mod, pix_size_mod])

    coordx_o_linspace=((np.arange(1.,res_object[0]+1.))-(1.+res_object[0])/2.)/res_object[0]*size_object[0]
    coordy_o_linspace=((np.arange(1.,res_object[1]+1.))-(1.+res_object[1])/2.)/res_object[1]*size_object[1]

    coordx_m_linspace=((np.arange(1.,res_modulator[0]+1.))-(1.+res_modulator[0])/2.)/res_modulator[0]*size_modulator[0]
    coordy_m_linspace=((np.arange(1.,res_modulator[1]+1.))-(1.+res_modulator[1])/2.)/res_modulator[1]*size_modulator[1]

    coordx_o, coordy_o=np.meshgrid(coordx_o_linspace, coordy_o_linspace)
    coordx_o=coordx_o.T
    coordy_o=coordy_o.T
    coordx_m, coordy_m=np.meshgrid(coordx_m_linspace, coordy_m_linspace)
    coordx_m=coordx_m.T
    coordy_m=coordy_m.T

    coordx_o_kron=np.kron(np.ones(res_modulator.astype('int').tolist()), coordx_o)
    coordy_o_kron=np.kron(np.ones(res_modulator.astype('int').tolist()), coordy_o)
    coordx_m_kron=np.kron(coordx_m, np.ones(res_object.astype('int').tolist()))
    coordy_m_kron=np.kron(coordy_m, np.ones(res_object.astype('int').tolist()))
#     pdb.set_trace()
    R_o2m=np.sqrt((coordx_o_kron-coordx_m_kron)**2.+(coordy_o_kron-coordy_m_kron)**2.+dist_obj2mod**2.)

    # for test
    deltaS_m=1
    deltaS_o=1
    deltaS_d=1
    
    theta_o2m=np.arctan(np.sqrt((coordx_o_kron-coordx_m_kron)**2.+(coordy_o_kron-coordy_m_kron)**2.)/dist_obj2mod)
#     propagator1_kron=np.exp(1j*2.*math.pi/mylambda*R_o2m)*deltaS_o*deltaS_m*np.cos(theta_o2m)/2.
#     pdb.set_trace()
    propagator1_kron_real=tf.cos(2.*math.pi/mylambda*R_o2m)*deltaS_o*deltaS_m*np.cos(theta_o2m)/2.
    propagator1_kron_imag=tf.sin(2.*math.pi/mylambda*R_o2m)*deltaS_o*deltaS_m*np.cos(theta_o2m)/2.
    
    dimx1=res_object[0]
    dimy1=res_object[1]
    dimx2=res_modulator[0]
    dimy2=res_modulator[1]

    propagator1_4d_yyxx_real=tf.reshape(propagator1_kron_real,[dimx2,dimx1,dimy2,dimy1])
    propagator1_4d_yxyx_real=tf.transpose(propagator1_4d_yyxx_real,(0,2,1,3))
    propagator1_2d_real=tf.reshape(propagator1_4d_yxyx_real,[dimy2*dimx2,dimy1*dimx1])

    propagator1_4d_yyxx_imag=tf.reshape(propagator1_kron_imag,[dimx2,dimx1,dimy2,dimy1])
    propagator1_4d_yxyx_imag=tf.transpose(propagator1_4d_yyxx_imag,(0,2,1,3))
    propagator1_2d_imag=tf.reshape(propagator1_4d_yxyx_imag,[dimy2*dimx2,dimy1*dimx1])
    
    return tf.cast(propagator1_2d_real, tf.float32), tf.cast(propagator1_2d_imag, tf.float32)

def lens_initialize(dim_mod, focal_lens, len_modulator):
    import tensorflow as tf
    import numpy as np
    import math
    import cmath
    import pdb
    # make sure there are points at the edge
    coordx = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator * dim_mod / (dim_mod - 1);
    coordy = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator * dim_mod / (dim_mod - 1);
    #coordx = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator;
    #coordy = (np.arange(1, dim_mod + 1) - (1 + dim_mod) / 2.) / dim_mod * len_modulator;
    coordx_matrix, coordy_matrix = np.meshgrid(coordx, coordy);
    coordx_matrix = coordx_matrix.T
    coordy_matrix = coordy_matrix.T
    coordr_matrix = np.sqrt(coordx_matrix**2. + coordy_matrix**2.)
#     amplitude_matrix = coordr_matrix<len_modulator/2;
    distance_phase_matrix = ((len_modulator * np.sqrt(2.) / 2.)**2. - coordx_matrix**2. - coordy_matrix**2.) / 2. / focal_lens;
    distance_phase_matrix = np.reshape(distance_phase_matrix, [dim_mod**2, 1])
    #print(coordx/len_modulator)
    #pdb.set_trace()
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
    train_points[:,:,0] = coordx_matrix 
    train_points[:,:,1] = coordy_matrix 
    train_points1 = np.reshape(train_points, [1, dim_mod * dim_mod, 2])
    
    coordx_query = (np.arange(1, dim_query + 1) - (1 + dim_query) / 2.) / dim_query * len_modulator
    coordy_query = (np.arange(1, dim_query + 1) - (1 + dim_query) / 2.) / dim_query * len_modulator    
    coordx_query_matrix, coordy_query_matrix = np.meshgrid(coordx_query, coordy_query);
    coordx_query_matrix = coordx_query_matrix.T
    coordy_query_matrix = coordy_query_matrix.T 
    query_points = np.zeros([dim_query, dim_query, 2])
    query_points[:,:,0] = coordx_query_matrix 
    query_points[:,:,1] = coordy_query_matrix 
    query_points1 = np.reshape(query_points, [1, dim_query * dim_query, 2])
    
    coordr_matrix = np.sqrt(coordx_matrix**2. + coordy_matrix**2.)
#     amplitude_matrix = coordr_matrix<len_modulator/2;
    distance_phase_matrix = ((len_modulator * np.sqrt(2.) / 2.)**2. - coordx_matrix**2. - coordy_matrix**2.) / 2. / focal_lens;
    distance_phase_matrix = np.reshape(distance_phase_matrix, [1, dim_mod**2, 1])
#     pdb.set_trace()
    return distance_phase_matrix, train_points1, query_points1

# in order to add a mask on the modulator during the training, which means zero on the mask
# square mask
def mask_modulator(dim_mod,dim_hole):
    import numpy as np
    if dim_hole >= dim_mod:
        dim_hole = dim_mod
    index_start = np.ceil((dim_mod - dim_hole) / 2.).astype('int32')
    index_end = np.floor((dim_mod + dim_hole) / 2.).astype('int32')
    mask = np.zeros([dim_mod, dim_mod])
    index = np.arange(index_start, index_end)
    mask[np.ix_(index,index)] = 1.
    return mask.astype('float32')

# round mask
def mask_modulator_round(dim_mod,dim_hole):
    import numpy as np
    coord_x = np.arange(0,dim_mod) - (dim_mod - 1.) / 2.
    coord_y = np.arange(0,dim_mod) - (dim_mod - 1.) / 2.
    coord_x_matrix, coord_y_matrix = np.meshgrid(coord_x, coord_y)
    coord_x_matrix = coord_x_matrix.T
    coord_y_matrix = coord_y_matrix.T
    coord_r_matrix = np.sqrt(coord_x_matrix**2. + coord_y_matrix**2.)
    index_matrix = coord_r_matrix < dim_hole / 2. * np.sqrt(2.)
    mask = np.zeros([dim_mod, dim_mod])
    mask[index_matrix] = 1.
    return mask.astype('float32')

def data_generator(batch_size, data, data_label):
    import numpy as np
    while(True):
        indexes = np.arange(0,data.shape[0],1)
        np.random.shuffle(indexes)
        max_range = int(data.shape[0]/batch_size)
        for i in range(max_range):
            data_temp = np.array([data[k] for k in indexes[i*batch_size:(i+1)*batch_size]])
            data_label_temp = np.array([data_label[k] for k in indexes[i*batch_size:(i+1)*batch_size]])
            yield data_temp, data_label_temp
