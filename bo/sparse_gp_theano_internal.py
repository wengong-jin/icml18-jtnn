
import theano
import theano.tensor as T

import numpy as np

from gauss import * 

from theano.tensor.slinalg import Cholesky as MatrixChol

import math

def n_pdf(x):
    return 1.0 / T.sqrt(2 * math.pi) * T.exp(-0.5 * x**2)

def log_n_pdf(x):
    return -0.5 * T.log(2 * math.pi) - 0.5 * x**2

def n_cdf(x):
    return 0.5 * (1.0 + T.erf(x / T.sqrt(2.0)))

def log_n_cdf_approx(x):
    return log_n_pdf(x) - T.log(-x - 1/x + 2 / x**3)

def log_n_cdf(x):
    x = T.switch(T.lt(x, casting(-10)), log_n_cdf_approx(x), T.log(n_cdf(x)))
    return x

def ratio(x):
    x = T.switch(T.lt(x, casting(-10)), -(casting(1.0)/x - casting(1.0)/x**3 + casting(3.0)/x**5 - casting(15.0)/x**7), n_cdf(x) / n_pdf(x))
    return x

def LogSumExp(x, axis = None):
    x_max = T.max(x, axis = axis, keepdims = True)
    return T.log(T.sum(T.exp(x - x_max), axis = axis, keepdims = True)) + x_max

##
# This class represents a GP node in the network
#

class Sparse_GP: 

    # n_points are the total number of training points (that is used for cavity computation)

    def __init__(self, n_inducing_points, n_points, input_d, input_means, input_vars, training_targets):

        self.ignore_variances = True
        self.n_inducing_points = n_inducing_points
        self.n_points = n_points
        self.input_d = input_d
        self.training_targets = training_targets
        self.input_means = input_means
        self.input_vars = input_vars

        # These are the actual parameters of the posterior distribution being optimzied
        # covCavity = (Kzz^-1 + LParamPost LParamPost^T * (n - 1) / n) and meanCavity = covCavity mParamPost * (n - 1) / n

        initial_value = np.zeros((n_inducing_points, n_inducing_points))
        self.LParamPost = theano.shared(value = initial_value.astype(theano.config.floatX), name = 'LParamPost', borrow = True)
        self.mParamPost = theano.shared(value = initial_value[ : , 0 : 1 ].astype(theano.config.floatX), name = 'mParamPost', borrow = True)
        self.lls = theano.shared(value = np.zeros(input_d).astype(theano.config.floatX), name = 'lls', borrow = True)
        self.lsf = theano.shared(value = np.zeros(1).astype(theano.config.floatX)[ 0 ], name = 'lsf', borrow = True)
        self.z = theano.shared(value = np.zeros((n_inducing_points, input_d)).astype(theano.config.floatX), name = 'z', borrow = True)
        self.lvar_noise = theano.shared(value = casting(0) * np.ones(1).astype(theano.config.floatX)[ 0 ], name = 'lvar_noise', borrow = True)

        self.set_for_training = casting(1.0)

        # We set the level of jitter to use  (added to the diagonal of Kzz)

        self.jitter = casting(1e-3)

    def compute_output(self):
        
        # We compute the output mean

        self.Kzz = compute_kernel(self.lls, self.lsf, self.z, self.z) + T.eye(self.z.shape[ 0 ]) * self.jitter * T.exp(self.lsf)
        self.KzzInv = T.nlinalg.MatrixInversePSD()(self.Kzz)
        LLt = T.dot(self.LParamPost, T.transpose(self.LParamPost))
        self.covCavityInv = self.KzzInv + LLt * casting(self.n_points - self.set_for_training) / casting(self.n_points)
        self.covCavity = T.nlinalg.MatrixInversePSD()(self.covCavityInv)
        self.meanCavity = T.dot(self.covCavity, casting(self.n_points - self.set_for_training) / casting(self.n_points) * self.mParamPost)
        self.KzzInvcovCavity = T.dot(self.KzzInv, self.covCavity)
        self.KzzInvmeanCavity = T.dot(self.KzzInv, self.meanCavity)
        self.covPosteriorInv = self.KzzInv + LLt
        self.covPosterior = T.nlinalg.MatrixInversePSD()(self.covPosteriorInv)
        self.meanPosterior = T.dot(self.covPosterior, self.mParamPost)
        self.Kxz = compute_kernel(self.lls, self.lsf, self.input_means, self.z)
        self.B = T.dot(self.KzzInvcovCavity, self.KzzInv) - self.KzzInv 
        v_out = T.exp(self.lsf) + T.dot(self.Kxz * T.dot(self.Kxz, self.B), T.ones_like(self.z[ : , 0 : 1 ]))

        if self.ignore_variances:

            self.output_means = T.dot(self.Kxz, self.KzzInvmeanCavity)
            self.output_vars = abs(v_out) + casting(0) * T.sum(self.input_vars)

        else:

            self.EKxz = compute_psi1(self.lls, self.lsf, self.input_means, self.input_vars, self.z)
            self.output_means = T.dot(self.EKxz, self.KzzInvmeanCavity)

            # In other layers we have to compute the expected variance

            self.B2 = T.outer(T.dot(self.KzzInv, self.meanCavity), T.dot(self.KzzInv, self.meanCavity))

            exact_output_vars = True

            if exact_output_vars:

                # We compute the exact output variance

                self.psi2 = compute_psi2(self.lls, self.lsf, self.z, self.input_means, self.input_vars)
                ll = T.transpose(self.EKxz[ :, None, : ] * self.EKxz[ : , : , None ], [ 1, 2, 0 ])
                kk = T.transpose(self.Kxz[ :, None, : ] * self.Kxz[ : , : , None ], [ 1, 2, 0 ])
                v1 = T.transpose(T.sum(T.sum(T.shape_padaxis(self.B2, 2) * (self.psi2 - ll), 0), 0, keepdims = True))
                v2 = T.transpose(T.sum(T.sum(T.shape_padaxis(self.B, 2) * (self.psi2 - kk), 0), 0, keepdims = True))

            else:

                # We compute the approximate output variance using the unscented kalman filter

                v1 = 0
                v2 = 0

                n = self.input_d
                for j in range(1, n + 1):
                    mask = T.zeros_like(self.input_vars)
                    mask = T.set_subtensor(mask[ :, j - 1 ] , 1)
                    inc = mask * T.sqrt(casting(n) * self.input_vars)
                    self.kplus = T.sqrt(casting(1.0) / casting(2 * n)) * compute_kernel(self.lls, self.lsf, self.input_means + inc, self.z)
                    self.kminus = T.sqrt(casting(1.0) / casting(2 * n)) * compute_kernel(self.lls, self.lsf, self.input_means - inc, self.z)

                    v1 += T.dot(self.kplus * T.dot(self.kplus, self.B2), T.ones_like(self.z[ : , 0 : 1 ]))
                    v1 += T.dot(self.kminus * T.dot(self.kminus, self.B2), T.ones_like(self.z[ : , 0 : 1 ]))
                    v2 += T.dot(self.kplus * T.dot(self.kplus, self.B), T.ones_like(self.z[ : , 0 : 1 ]))
                    v2 += T.dot(self.kminus * T.dot(self.kminus, self.B), T.ones_like(self.z[ : , 0 : 1 ]))

                v1 -= T.dot(self.EKxz * T.dot(self.EKxz, self.B2), T.ones_like(self.z[ : , 0 : 1 ]))
                v2 -= T.dot(self.Kxz * T.dot(self.Kxz, self.B), T.ones_like(self.z[ : , 0 : 1 ]))

            self.output_vars = abs(v_out) + abs(v2) + abs(v1)

        self.output_vars = self.output_vars + T.exp(self.lvar_noise)

        return

    def get_params(self):

        return [ self.lls, self.lsf, self.z, self.mParamPost, self.LParamPost, self.lvar_noise ]

    def set_params(self, params):

        self.lls.set_value(params[ 0 ])
        self.lsf.set_value(params[ 1 ])
        self.z.set_value(params[ 2 ])
        self.mParamPost.set_value(params[ 3 ])
        self.LParamPost.set_value(params[ 4 ])
        self.lvar_noise.set_value(params[ 5 ])
        
    ##
    # The next functions compute the log normalizer of each distribution (needed for energy computation)
    #

    def getLogNormalizerCavity(self):

        assert self.covCavity is not None  and self.meanCavity is not None and self.covCavityInv is not None 

        return casting(0.5 * self.n_inducing_points * np.log(2 * np.pi)) + casting(0.5) * T.nlinalg.LogDetPSD()(self.covCavity) + \
            casting(0.5) * T.dot(T.dot(T.transpose(self.meanCavity), self.covCavityInv), self.meanCavity)

    def getLogNormalizerPrior(self):

        assert self.KzzInv is not None 

        return casting(0.5 * self.n_inducing_points * np.log(2 * np.pi)) - casting(0.5) * T.nlinalg.LogDetPSD()(self.KzzInv)

    def getLogNormalizerPosterior(self):

        assert self.covPosterior is not None and self.meanPosterior is not None and self.covPosteriorInv is not None

        return casting(0.5 * self.n_inducing_points * np.log(2 * np.pi)) + casting(0.5) * T.nlinalg.LogDetPSD()(self.covPosterior) + \
            casting(0.5) * T.dot(T.dot(T.transpose(self.meanPosterior), self.covPosteriorInv), self.meanPosterior)

    ##
    # We return the contribution to the energy of the node (See last Eq. of Sec. 4 in http://arxiv.org/pdf/1602.04133.pdf v1)
    # 

    def getContributionToEnergy(self):

        assert self.n_points is not None and self.covCavity is not None and self.covPosterior is not None and self.input_means is not None

        logZpost = self.getLogNormalizerPosterior()
        logZprior = self.getLogNormalizerPrior()
        logZcav = self.getLogNormalizerCavity()

        # We multiply by the minibatch size and normalize terms according to the total number of points (n_points)

        return ((logZcav - logZpost) + logZpost / casting(self.n_points) - logZprior / casting(self.n_points)) * \
		T.cast(self.input_means.shape[ 0 ], 'float32') + T.sum(self.getLogZ())

	# These methods sets the inducing points to be a random subset of the inputs (we should receive more
	# inputs than inducing points), the length scales are set to the mean of the euclidean distance

    def initialize(self):

        input_means = np.array(theano.function([], self.input_means)())

        assert input_means.shape[ 0 ] >= self.n_inducing_points

        selected_points = np.random.choice(input_means.shape[ 0 ], self.n_inducing_points, replace = False)
        z = input_means[ selected_points, : ]

        # If we are not in the first layer, we initialize the length scales to one

        lls = np.zeros(input_means.shape[ 1 ])

        M = np.outer(np.sum(input_means**2, 1), np.ones(input_means.shape[ 0 ]))
        dist = M - 2 * np.dot(input_means, input_means.T) + M.T
        lls = np.log(0.5 * (np.median(dist[ np.triu_indices(input_means.shape[ 0 ], 1) ]) + 1e-3)) * np.ones(input_means.shape[ 1 ])
        
        self.lls.set_value(lls.astype(theano.config.floatX))
        self.z.set_value(z.astype(theano.config.floatX))
        self.lsf.set_value(np.zeros(1).astype(theano.config.floatX)[ 0 ])

        # We initialize the cavity and the posterior approximation to the prior but with a small random
        # mean so that the outputs are not equal to zero (otherwise the output of the gp will be zero and
        # the next layer will be initialized improperly).

        # If we are not in the first layer, we reduce the variance of the L and m

        L = np.random.normal(size = (self.n_inducing_points, self.n_inducing_points)) * 1.0
        m = self.training_targets.get_value()[ selected_points, : ]

        self.LParamPost.set_value(L.astype(theano.config.floatX))
        self.mParamPost.set_value(m.astype(theano.config.floatX))

    # This sets the node for prediction. It basically switches the cavity distribution to be the posterior approximation
    # Once set in this state the network cannot be trained any more.

    def setForPrediction(self):

        if self.set_for_training == casting(1.0):

            self.set_for_training = casting(0.0)

    # This function undoes the work done by the previous method

    def setForTraining(self):

        # We only do something if the node was set for prediction instead of training
                
        if self.set_for_training == casting(0.0):

            self.set_for_training == casting(1.0)

    def getLogZ(self):

        return -0.5 * T.log(2 * np.pi * self.output_vars) - 0.5 * (self.training_targets - self.output_means)**2 / self.output_vars

    def getPredictedValues(self):

        return self.output_means, self.output_vars

    def get_training_targets(self):
        return self.training_targets

    def set_training_targets(self, training_targets):
        self.training_targets = training_targets

    def compute_log_ei(self, x, incumbent):

        Kzz = compute_kernel(self.lls, self.lsf, self.z, self.z) + T.eye(self.z.shape[ 0 ]) * self.jitter * T.exp(self.lsf)
        KzzInv = T.nlinalg.MatrixInversePSD()(Kzz)
        LLt = T.dot(self.LParamPost, T.transpose(self.LParamPost))
        covCavityInv = KzzInv + LLt * casting(self.n_points - self.set_for_training) / casting(self.n_points)
        covCavity = T.nlinalg.MatrixInversePSD()(covCavityInv)
        meanCavity = T.dot(covCavity, casting(self.n_points - self.set_for_training) / casting(self.n_points) * self.mParamPost)
        KzzInvcovCavity = T.dot(KzzInv, covCavity)
        KzzInvmeanCavity = T.dot(KzzInv, meanCavity)
        Kxz = compute_kernel(self.lls, self.lsf, x, self.z)
        B = T.dot(KzzInvcovCavity, KzzInv) - KzzInv 
        v_out = T.exp(self.lsf) + T.dot(Kxz * T.dot(Kxz, B), T.ones_like(self.z[ : , 0 : 1 ])) # + T.exp(self.lvar_noise)
        m_out = T.dot(Kxz, KzzInvmeanCavity)
        s = (incumbent - m_out) / T.sqrt(v_out)

        log_ei = T.log((incumbent - m_out) * ratio(s) + T.sqrt(v_out)) + log_n_pdf(s)

        return log_ei

    def compute_log_averaged_ei(self, x, X, randomness, incumbent):

        # We compute the old predictive mean at x
        
        Kzz = compute_kernel(self.lls, self.lsf, self.z, self.z) + T.eye(self.z.shape[ 0 ]) * self.jitter * T.exp(self.lsf)
        KzzInv = T.nlinalg.MatrixInversePSD()(Kzz)
        LLt = T.dot(self.LParamPost, T.transpose(self.LParamPost))
        covCavityInv = KzzInv + LLt * casting(self.n_points - self.set_for_training) / casting(self.n_points)
        covCavity = T.nlinalg.MatrixInversePSD()(covCavityInv)
        meanCavity = T.dot(covCavity, casting(self.n_points - self.set_for_training) / casting(self.n_points) * self.mParamPost)
        KzzInvmeanCavity = T.dot(KzzInv, meanCavity)
        Kxz = compute_kernel(self.lls, self.lsf, x, self.z)
        m_old_x = T.dot(Kxz, KzzInvmeanCavity)

        # We compute the old predictive mean at X

        KXz = compute_kernel(self.lls, self.lsf, X, self.z)
        m_old_X = T.dot(KXz, KzzInvmeanCavity)

        # We compute the required cross covariance matrices

        KXX = compute_kernel(self.lls, self.lsf, X, X) - T.dot(T.dot(KXz, KzzInv), KXz.T) + T.eye(X.shape[ 0 ]) * self.jitter * T.exp(self.lsf)
        KXXInv = T.nlinalg.MatrixInversePSD()(KXX)

        KxX = compute_kernel(self.lls, self.lsf, x, X)
        xX = T.concatenate([ x, X ], 0)
        KxXz = compute_kernel(self.lls, self.lsf, xX, self.z)
        KxX = KxX - T.dot(T.dot(KxXz[ 0 : x.shape[ 0], : ], KzzInv), KxXz[ x.shape[ 0 ] : xX.shape[ 0 ], : ].T)

        # We compute the new posterior mean

        samples_internal = T.dot(MatrixChol()(KXX), randomness)

        new_predictive_mean = T.tile(m_old_x, [ 1, randomness.shape[ 1 ] ]) + T.dot(KxX, T.dot(KXXInv, samples_internal))

        # We compute the new posterior variance

        z_expanded = T.concatenate([ self.z, X ], 0)
        Kxz_expanded = compute_kernel(self.lls, self.lsf, x, z_expanded)
        Kzz_expanded = compute_kernel(self.lls, self.lsf, z_expanded, z_expanded) + T.eye(z_expanded.shape[ 0 ]) * self.jitter * T.exp(self.lsf)
        Kzz_expandedInv = T.nlinalg.MatrixInversePSD()(Kzz_expanded)
        v_out = T.exp(self.lsf) - T.dot(Kxz_expanded * T.dot(Kxz_expanded, Kzz_expandedInv), T.ones_like(z_expanded[ : , 0 : 1 ]))
        new_predictive_var = T.tile(v_out, [ 1, randomness.shape[ 1 ] ])

        s = (incumbent - new_predictive_mean) / T.sqrt(new_predictive_var)

        log_ei = T.log((incumbent - new_predictive_mean) * ratio(s) + T.sqrt(new_predictive_var)) + log_n_pdf(s)

        return T.mean(LogSumExp(log_ei, 1), 1)
