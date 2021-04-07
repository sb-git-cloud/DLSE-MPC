import sys
import cvxpy as cp
import tensorflow as tf
import numpy as np

def mpc(state_k, dsle_net, u0, **kwargs):
    assert u0.ndim == 1 and state_k.ndim == 1

    x0 = np.concatenate(state_k, u0, axis=0)
    # Assume u0 = [u_0^{(1)} u_0^{(2)} ... u_{horizon-1}^{(m)}]^T

    # Adjust constraints as optimization will be over [state_k; u0]

    # linear equality constraints combined with state_k != x0
    if 'Aeq' in kwargs:
        assert 'beq' in kwargs
        # Equality constraint for state_k
        aeq = np.concatenate((np.identity(len(state_k)), np.zeros(len(state_k), len(u0))), axis=1)
        beq = state_k

        # Equality constraint for input defined by user, adapted to DSLEA with x = [state_k;u0]
        kwargs['Aeq'] = np.concatenate((np.zeros(len(kwargs['beq']), len(state_k)), kwargs['Aeq']), axis=1)
        kwargs['Aeq'] = np.concatenate((aeq, kwargs['Aeq']), axis=0)
        kwargs['beq'] = np.concatenate((beq, kwargs['beq']), axis=0)
    else:
        kwargs['Aeq'] = np.concatenate((np.identity(len(state_k)), np.zeros(len(state_k), len(u0))), axis=1)
        kwargs['beq'] = state_k

    # linear inequality constraints
    if 'A' in kwargs:
        assert 'b' in kwargs
        kwargs['A'] = np.concatenate((np.zeros(len(kwargs['b']), len(state_k)), kwargs['A']), axis=1)

    # lower bound
    if 'lb' in kwargs:
        kwargs['lb'] = np.concatenate((np.zeros(len(state_k), 1), kwargs['lb']), axis=0)

    # upper bound
    if 'ub' in kwargs:
        kwargs['ub'] = np.concatenate((np.zeros(len(state_k), 1), kwargs['ub']), axis=0)

    x, y = dslea(x0, dsle_net, kwargs)
    return x[len(state_k):]

def dslea(x0, dsle_net, **kwargs):
    delta = sys.float_info.max
    constr = []
    x_cpvar = cp.Variable(x0.size)

    # Set stopping criterion parameter
    if not ('delta_tol' in kwargs):
        delta_tol = 1e-9
    else:
        delta_tol = kwargs['delta_tol']

    # Constraints:
    # linear inequality constraints
    if 'A' in kwargs:
        assert 'b' in kwargs
        constr += kwargs['A'] * x_cpvar - kwargs['b'] <= 0

    # linear equality constraints
    if 'Aeq' in kwargs:
        assert 'beq' in kwargs
        constr += kwargs['Aeq'] * x_cpvar - kwargs['beq'] == 0

    # lower bound
    if 'lb' in kwargs:
        constr += x_cpvar - kwargs['lb'] >= 0

    # upper bound
    if 'ub' in kwargs:
        constr += x_cpvar - kwargs['ub'] <= 0

    # Start optimization via cvx
    xk = x0
    while delta > delta_tol:
        yk = dh(xk, dsle_net)  # step 2 if DSLEA (solution of dual problem)
        cost = gstar(x_cpvar, yk, dsle_net)  # define cost
        problem = cp.Problem(cp.Minimize(cost), constr)  # set up problem
        problem.solve()  # step 3 solve convex problem
        delta = np.linalg.norm(xk-x_cpvar.value)/(1+np.linalg.norm(xk))

    return xk, yk

def dh(x, dsle_net):

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

    var_in_inverse = np.diag(np.reciprocal(dsle_net.var_in))
    vec_exp = np.exp(np.matmul(np.matmul(np.transpose(dsle_net.kWeightsBtm), var_in_inverse), (x - dsle_net.mean_in))
                     + np.transpose(dsle_net.kBiasBtm))

    # The return value uses this vector to
    #   a) compute the numerator by the matrix multiplication
    #
    #       [w_btm[:, 1] w_btm[:, 2] ... w_btm[:, K]] * [vec_exp[1] vec_exp[2] ... vec_exp[K]]^T
    #
    #       and an additional scaling via diag(var_in).
    #
    #   b) compute the denominator as the sum over vec_exp and the scaling factor of the output.The offset of the
    #      output is absent as this is the derivative of h.

    return np.matmul(np.matmul(var_in_inverse, dsle_net.kWeightsBtm), vec_exp) / (np.sum(vec_exp) * dsle_net.var_out)

def gstar(x, dh, dsle_net):

    # This corresponds to Step 3 of Algprithm 1 DSLEA and corresponds to the primal problem with linearized
    # concave part. See detailed comments for computation in function dh.

    var_in_inverse = np.diag(np.reciprocal(dsle_net.var_in))
    vec_exp = np.exp(np.matmul(np.matmul(np.transpose(dsle_net.kWeightsTop), var_in_inverse),  (x - dsle_net.mean_in))
                     + np.transpose(dsle_net.kBiasTop))

    return np.log(np.sum(vec_exp))-dsle_net.mean_out/dsle_net.var_out-x.dot(dh)
