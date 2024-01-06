import jax.numpy as np
from jax import vmap
import math
from jax.scipy.linalg import cho_factor, cho_solve
import numpy as nnp

LOG2PI = math.log(2 * math.pi)
INV2PI = (2 * math.pi) ** -1

def tile_arrays(listOfArrays: np.ndarray, num_latent):
    listOfArrays = [np.tile(arr, [num_latent, 1, 1]) for arr in listOfArrays]
    return listOfArrays

def make_diag(var):
    fill_diag = lambda x: np.diag(x)
    return vmap(fill_diag)(var[:,:,0])

@vmap
def _gaussian_expected_log_lik(y, post_mean, post_cov, var):
    # post_mean = post_mean.reshape(-1, 1)
    # post_cov = post_cov.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    # var = var.reshape(-1, 1)
    # version which computes sum and outputs scalar
    # exp_log_lik = (
    #     -0.5 * y.shape[-2] * np.log(2 * np.pi)  # multiplier based on dimensions needed if taking sum of other terms
    #     - 0.5 * np.sum(np.log(var))
    #     - 0.5 * np.sum(((y - post_mean) ** 2 + post_cov) / var)
    # )
    # version which computes individual parts and outputs vector
    # add some jitter for stability
    exp_log_lik = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(var)
        - 0.5 * ((y - post_mean) ** 2 + post_cov) / var
    )
    return exp_log_lik

def gaussian_expected_log_lik_diag(y, post_mean, post_cov, var):
    """
    Computes the "variational expectation", i.e. the
    expected log-likelihood, and its derivatives w.r.t. the posterior mean
        E[log 𝓝(yₙ|fₙ,σ²)] = ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
    :param y: data / observation (yₙ)
    :param post_mean: posterior mean (mₙ)
    :param post_cov: posterior variance (vₙ)
    :param var: variance, σ², of the Gaussian observation model p(yₙ|fₙ)=𝓝(yₙ|fₙ,σ²)
    :return:
        exp_log_lik: the expected log likelihood, E[log 𝓝(yₙ|fₙ,var)]  [scalar]
    """
    # post_cov = np.diag(post_cov)
    # var = np.diag(var)
    # var_exp = vmap(_gaussian_expected_log_lik)(y, post_mean, post_cov, var)
    var_exp = np.sum(_gaussian_expected_log_lik(y, post_mean, post_cov, var))
    # return np.sum(var_exp)
    return var_exp

def rotation_matrix(dt, omega):
    """
    Discrete time rotation matrix
    :param dt: step size [1]
    :param omega: frequency [1]
    :return:
        R: rotation matrix [2, 2]
    """
    R = np.array([
        [np.cos(omega * dt), -np.sin(omega * dt)],
        [np.sin(omega * dt),  np.cos(omega * dt)]
    ])
    return R

def broadcasting_elementwise(op, a, b):
    """
    Adapted from GPflow: https://github.com/GPflow/GPflow

    Apply binary operation `op` to every pair in tensors `a` and `b`.

    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    :param a: tf.Tensor, shape [n_1, ..., n_a]
    :param b: tf.Tensor, shape [m_1, ..., m_b]
    :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    """
    flatres = op(np.reshape(a, [-1, 1]), np.reshape(b, [1, -1]))
    return flatres.reshape(a.shape[0], b.shape[0])


def square_distance(X, X2):
    """
    Adapted from GPflow: https://github.com/GPflow/GPflow

    Returns ||X - X2ᵀ||²
    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.

    This function can deal with leading dimensions in X and X2.
    In the sample case, where X and X2 are both 2 dimensional,
    for example, X is [N, D] and X2 is [M, D], then a tensor of shape
    [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D]
    then the output will be [N1, S1, N2, S2].
    """
    Xs = np.sum(np.square(X), axis=-1)
    X2s = np.sum(np.square(X2), axis=-1)
    dist = -2 * np.tensordot(X, X2, [[-1], [-1]])
    dist += broadcasting_elementwise(np.add, Xs, X2s)
    return dist


def scaled_squared_euclid_dist(X, X2, ell):
    """
    Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.
    Adapted from GPflow: https://github.com/GPflow/GPflow
    """
    return square_distance(X / ell, X2 / ell)


def softplus(x_):
    # return np.log(1.0 + np.exp(x_))
    return np.log(1. + np.exp(-np.abs(x_))) + np.maximum(x_, 1e-24)  # safer version (but derivatve can have issues)


def softplus_inv(x_):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    """
    if x_ is None:
        return x_
    else:
        # return np.log(np.exp(x_) - 1)
        return np.log(1. - np.exp(-np.abs(x_))) + np.maximum(x_, 1e-24)  # safer version


def solve(P, Q):
    """
    Compute P^-1 Q, where P is a PSD matrix, using the Cholesky factorisation
    """
    L = cho_factor(P, lower=True)
    return cho_solve(L, Q)


def transpose(P):
    return np.swapaxes(P, -1, -2)


def inv(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P, lower=True)
    return cho_solve(L, np.eye(P.shape[-1]))


def input_admin(t, y, r):
    """
    Order the inputs.
    :param t: training inputs [N, 1]
    :param y: observations at the training inputs [N, 1]
    :param r: training spatial inputs
    :return:
        t_train: training inputs [N, 1]
        y_train: training observations [N, R]
        r_train: training spatial inputs [N, R]
        dt_train: training step sizes, Δtₙ = tₙ - tₙ₋₁ [N, 1]
    """
    assert t.shape[0] == y.shape[0]
    if t.ndim < 2:
        t = nnp.expand_dims(t, 1)  # make 2-D
    if y.ndim < 2:
        y = nnp.expand_dims(y, 1)  # make 2-D
    if r is None:
        if t.shape[1] > 1:
            r = t[:, 1:]
            t = t[:, :1]
        else:
            r = nnp.nan * t  # np.empty((1,) + x.shape[1:]) * np.nan
    if r.ndim < 2:
        r = nnp.expand_dims(r, 1)  # make 2-D
    ind = nnp.argsort(t[:, 0], axis=0)
    t_train = t[ind, ...]
    y_train = y[ind, ...]
    r_train = r[ind, ...]
    dt_train = nnp.concatenate([np.array([0.0]), nnp.diff(t_train[:, 0])])
    return (
        np.array(t_train, dtype=np.float64),
        np.array(y_train, dtype=np.float64),
        np.array(r_train, dtype=np.float64),
        np.array(dt_train, dtype=np.float64),
    )


def compute_conditional_statistics(A_fwd, A_back, Pinf):
    """
    This version uses cho_factor and cho_solve - much more efficient when using JAX

    Predicts marginal states at new time points. (new time points should be sorted)
    Calculates the conditional density:
             p(xₙ|u₋, u₊) = 𝓝(Pₙ @ [u₋, u₊], Tₙ)

    :param x_test: time points to generate observations for [N]
    :param x: inducing state input locations [M]
    :param kernel: prior object providing access to state transition functions
    :param ind: an array containing the index of the inducing state to the left of every input [N]
    :return: parameters for the conditional mean and covariance
            P: [N, D, 2*D]
            T: [N, D, D]
    """
    # dt_fwd = x_test[..., 0] - x[ind, 0]
    # dt_back = x[ind + 1, 0] - x_test[..., 0]
    # A_fwd = kernel.state_transition(dt_fwd)
    # A_back = kernel.state_transition(dt_back)
    # Pinf = kernel.stationary_covariance()
    Q_fwd = Pinf - A_fwd @ Pinf @ A_fwd.T
    Q_back = Pinf - A_back @ Pinf @ A_back.T
    A_back_Q_fwd = A_back @ Q_fwd
    Q_mp = Q_back + A_back @ A_back_Q_fwd.T

    jitter = 1e-8 * np.eye(Q_mp.shape[0])
    chol_Q_mp = cho_factor(Q_mp + jitter, lower=True)
    Q_mp_inv_A_back = cho_solve(chol_Q_mp, A_back)  # V = Q₋₊⁻¹ Aₜ₊

    # The conditional_covariance T = Q₋ₜ - Q₋ₜAₜ₊ᵀQ₋₊⁻¹Aₜ₊Q₋ₜ == Q₋ₜ - Q₋ₜᵀAₜ₊ᵀL⁻ᵀL⁻¹Aₜ₊Q₋ₜ
    T = Q_fwd - A_back_Q_fwd.T @ Q_mp_inv_A_back @ Q_fwd
    # W = Q₋ₜAₜ₊ᵀQ₋₊⁻¹
    W = Q_fwd @ Q_mp_inv_A_back.T
    P = np.concatenate([A_fwd - W @ A_back @ A_fwd, W], axis=-1)
    return P, T


def predict_from_state(ind, post_mean, post_cov, gain, A_fwd, A_back, Pinf):
    """
    wrapper function to vectorise predict_at_t_()
    """
    predict_from_state_func = vmap(
        predict_from_state_, (0, None, None, None, 0, 0, None)
    )
    return predict_from_state_func(ind, post_mean, post_cov, gain, A_fwd, A_back, Pinf)


def predict_from_state_(ind, post_mean, post_cov, gain, A_fwd, A_back, Pinf):
    """
    predict the state distribution at time t by projecting from the neighbouring inducing states
    """
    
    P, T = compute_conditional_statistics(A_fwd, A_back, Pinf)
    # joint posterior (i.e. smoothed) mean and covariance of the states [u_, u+] at time t:
    mean_joint = np.block([[post_mean[ind]], [post_mean[ind + 1]]])
    cross_cov = gain[ind] @ post_cov[ind + 1]
    cov_joint = np.block([[post_cov[ind], cross_cov], [cross_cov.T, post_cov[ind + 1]]])
    return P @ mean_joint, P @ cov_joint @ P.T + T


def temporal_conditional(X, X_test, mean, cov, gain, A_fwd, A_back, Pinf):
    """
    predict from time X to time X_test give state mean and covariance at X
    """
    Pinf = Pinf[None, ...]
    minf = np.zeros([1, Pinf.shape[1], 1])
    mean_aug = np.concatenate([minf, mean, minf])
    cov_aug = np.concatenate([Pinf, cov, Pinf])
    gain = np.concatenate([np.zeros_like(gain[:1]), gain])

    # figure out which two training states each test point is located between
    ind_test = (
        np.searchsorted(
            X.reshape(
                -1,
            ),
            X_test.reshape(
                -1,
            ),
        )
        - 1
    )
    # project from training states to test locations
    test_mean, test_cov = predict_from_state(
        ind_test, mean_aug, cov_aug, gain, A_fwd, A_back, Pinf[0]
    )

    return test_mean, test_cov


def inv_vmap(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P, lower=True)
    return cho_solve(L, np.tile(np.eye(P.shape[-1]), [P.shape[0], 1, 1]))


def diag(P):
    """
    a broadcastable version of np.diag, for when P is size [N, D, D]
    """
    return np.diagonal(P, axis1=1, axis2=2)





def ensure_positive_precision(K):
    """
    Check whether matrix K has positive diagonal elements.
    If not, then replace the negative elements with default value 0.01
    """
    # K_diag = diag(diag(K))
    K_diag = vmap_diag(diag(K))
    K = np.where(np.any(diag(K) < 0), np.where(K_diag < 0, 1e-2, K_diag), K)
    return K


def mvn_logpdf(x, mean, cov, mask=None):
    """
    evaluate a multivariate Gaussian (log) pdf
    """
    x = x.reshape(-1, 1)
    mean = mean.reshape(-1, 1)
    if mask is not None:
        # build a mask for computing the log likelihood of a partially observed multivariate Gaussian
        maskv = mask.reshape(-1, 1)
        x = np.where(maskv, 0.0, x)
        mean = np.where(maskv, 0.0, mean)
        cov_masked = np.where(
            maskv + maskv.T, 0.0, cov
        )  # ensure masked entries are independent
        cov = np.where(
            np.diag(mask), INV2PI, cov_masked
        )  # ensure masked entries return log like of 0

    n = mean.shape[0]
    cho, low = cho_factor(cov, lower=True)
    log_det = 2 * np.sum(np.log(np.abs(np.diag(cho))))
    diff = x - mean
    scaled_diff = cho_solve((cho, low), diff)
    distance = diff.T @ scaled_diff
    return np.squeeze(-0.5 * (distance + n * LOG2PI + log_det))


@vmap
def vmap_diag(P):
    return np.diag(P)
