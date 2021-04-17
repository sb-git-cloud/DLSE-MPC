import sys
import cvxpy as cp
import tensorflow as tf
import numpy as np

class MpcProblem:
    def __init__(self, dsle_net, state_dim, inp_dim, **kwargs):

        self.state_dim = state_dim  # state dimension
        self.inp_dim = inp_dim  # input dimension
        self.x0_cppar = cp.Parameter(state_dim)  # optimization parameter

        # Adjust constraints as optimization will be over [state_k; u0]
        if 'Aeq' in kwargs:  # linear equality constraints: Aeq * u == beq, combined with state_k == x0
            assert 'beq' in kwargs
            # Equality constraint for state_k
            aeq = np.concatenate((np.identity(state_dim), np.zeros((state_dim, inp_dim))), axis=1)
            beq = self.x0_cppar

            # Equality constraint for input defined by user, adapted to DSLEA with x = [state_k;u0]
            kwargs['Aeq'] = np.concatenate((np.zeros((len(kwargs['beq']), state_dim)), kwargs['Aeq']), axis=1)
            kwargs['Aeq'] = np.concatenate((aeq, kwargs['Aeq']), axis=0)
            kwargs['beq'] = np.concatenate((beq, kwargs['beq']), axis=0)
        else:
            kwargs['Aeq'] = np.concatenate((np.identity(state_dim), np.zeros((state_dim, inp_dim))), axis=1)
            kwargs['beq'] = self.x0_cppar

        if 'A' in kwargs:  # linear inequality constraints: A u <= b
            assert 'b' in kwargs
            kwargs['A'] = np.concatenate((np.zeros((len(kwargs['b']), state_dim)), kwargs['A']), axis=1)

        if 'lb' in kwargs:  # lower bound: u >= lb
            kwargs['lb'] = np.concatenate((np.zeros((state_dim, 1), kwargs['lb'])), axis=0)

        if 'ub' in kwargs:  # upper bound: u <= ub
            kwargs['ub'] = np.concatenate((np.zeros((state_dim, 1), kwargs['ub'])), axis=0)

        self.dslea_problem = DsleaProblem(dsle_net, state_dim+inp_dim, **kwargs)

    def solve(self, state_k, u0, verbose=False, step_tol=1e-6, MAX_ITERS = 1e3, **kwargs):
        self.x0_cppar.value = state_k
        x0 = np.concatenate(state_k, u0, axis=0)
        xu, dh = self.dslea_problem.solve(x0, verbose, step_tol, MAX_ITERS, kwargs)
        return xu[self.state_dim:]  # return only the control sequence

class DsleaProblem:
    def __init__(self, dsle_net, n_inputs, **kwargs):

        # Extract network parameters from model for faster access during optimization
        # Preprocessing layers
        self.mean_in = tf.make_ndarray(tf.make_tensor_proto(dsle_net.get_layer('prepro_input').mean))  # input
        self.var_in = tf.make_ndarray(tf.make_tensor_proto(dsle_net.get_layer('prepro_input').variance))  # input
        self.mean_out = float(tf.make_ndarray(tf.make_tensor_proto(dsle_net.get_layer('prepro_output').mean)))  # output
        self.var_out = float(tf.make_ndarray(tf.make_tensor_proto(dsle_net.get_layer('prepro_output').variance)))  # out
        # Top Log-sum-exp layer
        self.kWeightsTop = dsle_net.get_layer('top').get_weights()[0]
        self.kBiasTop = dsle_net.get_layer('top').get_weights()[1]
        # Bottom log-sum-exp layer
        self.kWeightsBtm = dsle_net.get_layer('btm').get_weights()[0]
        self.kBiasBtm = dsle_net.get_layer('btm').get_weights()[1]

        # Prepare optimization problem fro CVX
        self.x_cpvar = cp.Variable(n_inputs)
        self.dh_cppar = cp.Parameter(n_inputs)

        # Constraints:
        self.constr = []
        if 'A' in kwargs:  # linear inequality constraints: A * x <= b
            assert 'b' in kwargs
            self.constr += [kwargs['A'] @ self.x_cpvar <= kwargs['b']]

        if 'Aeq' in kwargs:  # linear equality constraints:  Aeq * x == beq
            assert 'beq' in kwargs
            self.constr += [kwargs['Aeq'] @ self.x_cpvar == kwargs['beq']]

        if 'lb' in kwargs:  # lower bound: x >= lb
            self.constr += [self.x_cpvar >= kwargs['lb']]

        if 'ub' in kwargs:  # upper bound: x <= ub
            self.constr += [self.x_cpvar <= kwargs['ub']]

        # Initialize optimization problem
        self.delta = sys.float_info.max
        self.cost = self.gstar(self.x_cpvar, self.dh_cppar)  # define cost
        self.cp_problem = cp.Problem(cp.Minimize(self.cost), self.constr)  # set up problem

    def solve(self, x0, verbose=False, step_tol=1e-6, MAX_ITERS = 1e3, **kwargs):

        # Setup optimization problem
        step = sys.float_info.max
        xk = x0
        i = 0
        while step > step_tol and i < MAX_ITERS:
            self.dh_cppar.value = self.dh(xk)  # step 2 if DSLEA (solution of dual problem)
            self.cp_problem.solve(verbose=verbose)  # step 3 solve convex problem
            step = np.linalg.norm(xk-self.x_cpvar.value)/(1+np.linalg.norm(xk))
            i += 1

        return xk, self.dh_cppar.value

    def dh(self, x):

        # Computation following
        #   G.C.Calafiore, S.Gaubert, and C.Possieri, “A universal approximation result for difference of log - sum -
        #   exp neural networks,” IEEE Transactions on Neural Networks and Learning Systems, 2020.
        #   DOI: 10.1109 / TNNLS.2020.2975051.
        #
        # and
        #
        #   S. Br{\"u}ggemann and C. Possieri, "On the Use of Difference of Log-Sum-Exp Neural Networks to Solve
        #   Data-Driven Model Predictive Control Tracking Problems," in IEEE Control Systems Letters, vol. 5, no. 4,
        #   pp. 1267-1272, Oct. 2021, doi: 10.1109/LCSYS.2020.3032083.
        #
        # This function corresponds to Step 2 of Algorithm 1(DLSEA) in the later publication.
        #
        #
        # The following is the sum of the numerator with each term corresponding to one row in the vector, excluding the
        # vector multiplication, i.e.,
        #
        #   vec_exp[k] = exp(\gamma ^ {k} \chi) +\delta_k,
        #
        # where \gamma ^ {k} is the weight row vector for neuron k of the bottom layer (w_btm),  \chi the current
        # solution of the primal problem(x) and delta_k the bias of neuron k of the bottom layer(bias_btm).
        #
        # We additionally take into account the normalization of the input here via var_in_inverse and it_mean_in.

        var_in_inverse = np.diag(np.reciprocal(self.var_in))
        vec_exp = np.exp(np.matmul(np.matmul(np.transpose(self.kWeightsBtm), var_in_inverse), (x - self.mean_in))
                         + np.transpose(self.kBiasBtm))

        # The return value uses this vector to
        #   a) compute the numerator by the matrix multiplication
        #
        #       [w_btm[:, 1] w_btm[:, 2] ... w_btm[:, K]] * [vec_exp[1] vec_exp[2] ... vec_exp[K]]^T
        #
        #       and an additional scaling via diag(var_in).
        #
        #   b) compute the denominator as the sum over vec_exp and the scaling factor of the output.The offset of the
        #      output is absent as this is the derivative of h.

        return np.matmul(np.matmul(var_in_inverse, self.kWeightsBtm), vec_exp) / (np.sum(vec_exp) * self.var_out)

    def gstar(self, x, dh):

        # This corresponds to Step 3 of Algprithm 1 DSLEA and corresponds to the primal problem with linearized
        # concave part. See detailed comments for computation in function dh.
        #
        # Instead of numpy, we use expressions from cvxpy, which are equivalent

        var_in_inverse = cp.diag(cp.inv_pos(self.var_in))
        # vec_exp = cp.exp(cp.matmul(cp.matmul(cp.transpose(self.kWeightsTop), var_in_inverse), (x-self.mean_in))
        #                  + cp.transpose(self.kBiasTop))
        # return cp.log(cp.sum(vec_exp)) - self.mean_out / self.var_out - cp.transpose(x)@dh
        return cp.log_sum_exp(cp.matmul(cp.matmul(cp.transpose(self.kWeightsTop), var_in_inverse), (x-self.mean_in))
                              + cp.transpose(self.kBiasTop)) - self.mean_out / self.var_out - cp.transpose(x)@dh