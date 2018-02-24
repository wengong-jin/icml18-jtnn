
import theano
import theano.tensor as T

import numpy as np
from scipy.spatial.distance import cdist

def casting(x):
    return np.array(x).astype(theano.config.floatX)

def compute_kernel(lls, lsf, x, z):

    ls = T.exp(lls)
    sf = T.exp(lsf)

    if x.ndim == 1:
        x = x[ None, : ]

    if z.ndim == 1:
        z = z[ None, : ]

    lsre = T.outer(T.ones_like(x[ :, 0 ]), ls)

    r2 = T.outer(T.sum(x * x / lsre, 1), T.ones_like(z[ : , 0 : 1 ])) - np.float32(2) * \
        T.dot(x / lsre, T.transpose(z)) + T.dot(np.float32(1.0) / lsre, T.transpose(z)**2)

    k = sf * T.exp(-np.float32(0.5) * r2)

    return k

def compute_kernel_numpy(lls, lsf, x, z):

    ls = np.exp(lls)
    sf = np.exp(lsf)

    if x.ndim == 1:
        x= x[ None, : ]

    if z.ndim == 1:
        z= z[ None, : ]

    lsre = np.outer(np.ones(x.shape[ 0 ]), ls)

    r2 = np.outer(np.sum(x * x / lsre, 1), np.ones(z.shape[ 0 ])) - 2 * np.dot(x / lsre, z.T) + np.dot(1.0 / lsre, z.T **2)

    k = sf * np.exp(-0.5*r2)

    return k

##
# xmean and xvar can be vectors of input points
#
# This is the expected value of the kernel
#

def compute_psi1(lls, lsf, xmean, xvar, z):

    if xmean.ndim == 1:
        xmean = xmean[ None, : ]

    ls = T.exp(lls)
    sf = T.exp(lsf)
    lspxvar = ls + xvar
    constterm1 = ls / lspxvar
    constterm2 = T.prod(T.sqrt(constterm1), 1)
    r2_psi1 = T.outer(T.sum(xmean * xmean / lspxvar, 1), T.ones_like(z[ : , 0 : 1 ])) \
        - np.float32(2) * T.dot(xmean / lspxvar, T.transpose(z)) + \
        T.dot(np.float32(1.0) / lspxvar, T.transpose(z)**2)
    psi1 = sf * T.outer(constterm2, T.ones_like(z[ : , 0 : 1 ])) * T.exp(-np.float32(0.5) * r2_psi1)

    return psi1

def compute_psi1_numpy(lls, lsf, xmean, xvar, z):

    if xmean.ndim == 1:
        xmean = xmean[ None, : ]

    ls = np.exp(lls)
    sf = np.exp(lsf)
    lspxvar = ls + xvar
    constterm1 = ls / lspxvar
    constterm2 = np.prod(np.sqrt(constterm1), 1)
    r2_psi1 = np.outer(np.sum(xmean * xmean / lspxvar, 1), \
        np.ones(z.shape[ 0 ])) - 2 * np.dot(xmean / lspxvar, z.T) + \
        np.dot(1.0 / lspxvar, z.T **2)
    psi1 = sf * np.outer(constterm2, np.ones(z.shape[ 0 ])) * np.exp(-0.5 * r2_psi1)
    return psi1

def compute_psi2(lls, lsf, z, input_means, input_vars):

    ls = T.exp(lls)
    sf = T.exp(lsf)
    b = ls / casting(2.0)
    term_1 = T.prod(T.sqrt(b / (b + input_vars)), 1)

    scale = T.sqrt(4 * (2 * b[ None, : ] + 0 * input_vars))
    scaled_z = z[ None, : , : ] / scale[ : , None , : ]
    scaled_z_minus_m = scaled_z
    r2b = T.sum(scaled_z_minus_m**2, 2)[ :, None, : ] + T.sum(scaled_z_minus_m**2, 2)[ :, : , None ] - \
        2 * T.batched_dot(scaled_z_minus_m, np.transpose(scaled_z_minus_m, [ 0, 2, 1 ]))
    term_2 = T.exp(-r2b)

    scale = T.sqrt(4 * (2 * b[ None, : ] + 2 * input_vars))
    scaled_z = z[ None, : , : ] / scale[ : , None , : ]
    scaled_m = input_means / scale
    scaled_m = T.tile(scaled_m[ : , None, : ], [ 1, z.shape[ 0 ], 1])
    scaled_z_minus_m = scaled_z - scaled_m
    r2b = T.sum(scaled_z_minus_m**2, 2)[ :, None, : ] + T.sum(scaled_z_minus_m**2, 2)[ :, : , None ] + \
        2 * T.batched_dot(scaled_z_minus_m, np.transpose(scaled_z_minus_m, [ 0, 2, 1 ]))
    term_3 = T.exp(-r2b)
    
    psi2_computed = sf**casting(2.0) * term_1[ :, None, None ] * term_2 * term_3

    return T.transpose(psi2_computed, [ 1, 2, 0 ])

def compute_psi2_numpy(lls, lsf, z, input_means, input_vars):

    ls = np.exp(lls)
    sf = np.exp(lsf)
    b = ls / casting(2.0)
    term_1 = np.prod(np.sqrt(b / (b + input_vars)), 1)

    scale = np.sqrt(4 * (2 * b[ None, : ] + 0 * input_vars))
    scaled_z = z[ None, : , : ] / scale[ : , None , : ]
    scaled_z_minus_m = scaled_z
    r2b = np.sum(scaled_z_minus_m**2, 2)[ :, None, : ] + np.sum(scaled_z_minus_m**2, 2)[ :, : , None ] - \
        2 * np.einsum('ijk,ikl->ijl', scaled_z_minus_m, np.transpose(scaled_z_minus_m, [ 0, 2, 1 ]))
    term_2 = np.exp(-r2b)

    scale = np.sqrt(4 * (2 * b[ None, : ] + 2 * input_vars))
    scaled_z = z[ None, : , : ] / scale[ : , None , : ]
    scaled_m = input_means / scale
    scaled_m = np.tile(scaled_m[ : , None, : ], [ 1, z.shape[ 0 ], 1])
    scaled_z_minus_m = scaled_z - scaled_m
    r2b = np.sum(scaled_z_minus_m**2, 2)[ :, None, : ] + np.sum(scaled_z_minus_m**2, 2)[ :, : , None ] + \
        2 * np.einsum('ijk,ikl->ijl', scaled_z_minus_m, np.transpose(scaled_z_minus_m, [ 0, 2, 1 ]))
    term_3 = np.exp(-r2b)
    
    psi2_computed = sf**casting(2.0) * term_1[ :, None, None ] * term_2 * term_3
    psi2_computed = np.transpose(psi2_computed, [ 1, 2, 0 ])

    return psi2_computed
