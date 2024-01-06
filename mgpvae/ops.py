import jax.numpy as np
from jax import vmap
from mgpvae.util import mvn_logpdf, solve, transpose
from jax.lax import scan, cond
import math

INV2PI = (2 * math.pi) ** -1


@vmap
def vmap_mvn_logpdf(*args, **kwargs):
    return mvn_logpdf(*args, **kwargs)


def process_noise_covariance(A, Pinf):
    """See equation 3.11 https://aaltodoc.aalto.fi/bitstream/handle/123456789/19842/isbn9789526067117.pdf?sequence=1&isAllowed=y
    This is to obtain the Q_{i, i+1} matrix
    """
    Q = Pinf - A @ Pinf @ transpose(A)
    return Q


def kalman_filter_independent_latent(
    dt,
    kernel,
    y,
    noise_cov,
    mask=None,
    parallel=False,
    return_predict=False,
    spatiotemporal=False,
):
    """Entry point for the kalman filtering algorithm with multiple latent channels
    
    """

    # Obtain stationary covariance P_\infty, which is used to initialise the filter
    # s_0 ~ N(0, P_\infty)
    # Spatiotemporal shape: (num_latent, num_space, state_dim, state_dim2)
    # state_dim = 2 for matern32 kernel
    Pinf = kernel.stationary_covariance()
    # Compute the transitio matrices between each time point with dt
    # Spatiotemporal shape: (T, num_latent, num_space, state_dim, state_dim2)
    As = vmap(kernel.state_transition)(dt)
    # Compute the measurement matrix
    # for spatiotemporal=True, this is L_{ss} \otimes H^{(t)}
    # Spatiotemporal shape: (num_latent, num_space, num_space*state_dim)
    H = kernel.measurement_model()
    if spatiotemporal == True:
        # If spatiotemporal then need minf=0 to have dimensions
        # (num_latent, num_space, state_dim, 1)
        minf = np.zeros([Pinf.shape[0], Pinf.shape[1], Pinf.shape[2], 1])
        # This tuple is needed to create diagonal matrices later
        # e.g.
        # (DeviceArray([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,
        #       7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
        #      15, 15, 16, 16, 17, 17], dtype=int64), 
        # DeviceArray([ 0,  1,  0,  1,  2,  3,  2,  3,  4,  5,  4,  5,  6,  7,  6,
        #       7,  8,  9,  8,  9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15,
        #      14, 15, 16, 17, 16, 17], dtype=int64))
        block_index = kernel.block_index
    else:
        minf = np.zeros([Pinf.shape[0], Pinf.shape[1], 1])
        block_index = None

    # Batch over num_latent dimensions i.e. filter independent for each
    # latent channel. 
    # Spatiotemporal shapes:
    # As: (T, num_latent, num_space, state_dim, state_dim)
    # Pinf: (num_latent, num_space, state_dim, state_dim)
    # minf: (num_latent, num_space, state_dim, state_dim)
    # H: (num_latent, num_space, num_latent*num_space)
    # y: (T, num_latent, num_space)
    # noise_cov: (T, num_latent, num_space)
    ell, (means, covs) = vmap(
        kalman_filter, (1, 0, 0, 0, 1, 1, None, None, None, None, None)
    )(
        As,
        Pinf,
        minf,
        H,
        y,
        noise_cov,
        mask,
        block_index,
        parallel,
        return_predict,
        spatiotemporal,
    )
    return np.sum(ell), (means, covs)


def kalman_filter(
    As,
    Pinf,
    minf,
    H,
    y,
    noise_cov,
    mask=None,
    block_index=None,
    parallel=False,
    return_predict=False,
    spatiotemporal=False,
):
    """
    Run the Kalman filter to get p(fₙ|y₁,...,yₙ).
    Assumes a heteroscedastic Gaussian observation model, i.e. var is vector valued
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, D, 1]
    :param noise_cov: observation noise covariances [N, D, D]
    :param mask: boolean mask for the observations (to indicate missing data locations) [N, D, 1]
    :param parallel: flag to switch between parallel and sequential implementation of Kalman filter
    :param return_predict: flag whether to return predicted state, rather than updated state
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """
    if len(y.shape) == 2:
        y = np.expand_dims(y, axis=-1)
        noise_cov = np.expand_dims(noise_cov, axis=-1)
    if mask is not None:
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
    if mask is None:
        mask = np.zeros((y.shape[0], 1, 1), dtype=bool)
    else:
        if len(mask.shape) == 4:
            mask = 1 - mask[:, 0, :1, :1]
        elif len(mask.shape) == 3:
            mask = 1 - mask[:, :1, :1]
        else:
            raise ValueError("Mask has incorrect shape")

    # this function computes the Q_{i,i+1} matrix (using some clever trick)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    if parallel:
        raise NotImplementedError()
        # ell, means, covs = _parallel_kf(
        #     As, Qs, H, y, noise_cov, minf, Pinf, mask, return_predict=return_predict
        # )
    else:
        if spatiotemporal:
            # As: (T, num_space, state_dim, state_dim)
            # Qs: (T, num_space, state_dim, state_dim)
            # H: (num_space, num_space*state_dim)
            # y: (T, num_space, 1). This is the variational mean
            # noise_cov: (T, num_space, 1). This is the variational variational
            # minf: (num_space, state_dim, 1)
            # Pinf: (num_space, state_dim, state_dim)
            # mask: (T, 1, 1). All are False/0 values if we don't have data mask
            ell, means, covs = _sequential_kf_spatiotemporal(
                As, Qs, H, y, noise_cov, minf, Pinf, mask, block_index
            )
        else:
            ell, means, covs = _sequential_kf(
                As, Qs, H, y, noise_cov, minf, Pinf, mask, return_predict=return_predict
            )
    return ell, (means, covs)


def rauch_tung_striebel_smoother_independent_latent(
    dt,
    kernel,
    filter_mean,
    filter_cov,
    mask=None,
    return_full=False,
    parallel=False,
    spatiotemporal=False,
):
    """Entry point for the kalman smoothing algorithm with multiple latent channels
    """

    # Obtain stationary covariance P_\infty, which is used to initialise the filter
    # s_0 ~ N(0, P_\infty)
    # Spatiotemporal shape: (num_latent, num_space, state_dim, state_dim2)
    # state_dim = 2 for matern32 kernel
    Pinf = kernel.stationary_covariance()
    # Compute the transitio matrices between each time point with dt
    # Spatiotemporal shape: (T, num_latent, num_space, state_dim, state_dim2)
    As = vmap(kernel.state_transition)(dt)
    # Compute the measurement matrix
    # for spatiotemporal=True, this is L_{ss} \otimes H^{(t)}
    # Spatiotemporal shape: (num_latent, num_space, num_space*state_dim)
    H = kernel.measurement_model()

    if spatiotemporal == True:
        # This tuple is needed to create block diagonal matrices later
        # e.g.
        # (DeviceArray([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,
        #       7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
        #      15, 15, 16, 16, 17, 17], dtype=int64), 
        # DeviceArray([ 0,  1,  0,  1,  2,  3,  2,  3,  4,  5,  4,  5,  6,  7,  6,
        #       7,  8,  9,  8,  9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15,
        #      14, 15, 16, 17, 16, 17], dtype=int64))
        block_index = kernel.block_index
    else:
        block_index = None
    # Batch over num_latent dimensions i.e. smooth independently for each
    # latent channel. 
    # Spatiotemporal shapes:
    # As: (T, num_latent, num_space, state_dim, state_dim)
    # Pinf: (num_latent, num_space, state_dim, state_dim)
    # H: (num_latent, num_space, num_latent*num_space)
    # filter_mean: (num_latent, T, num_space, state_dim, 1)
    # filter_cov: (num_latent, T, num_space, state_dim, state_dim)
    means, covs, gains = vmap(
        rauch_tung_striebel_smoother, (1, 0, 0, 0, 0, None, None, None, None, None)
    )(
        As,
        Pinf,
        H,
        filter_mean,
        filter_cov,
        mask,
        block_index,
        return_full,
        parallel,
        spatiotemporal,
    )
    if return_full:
        # typically no batching
        # Only return state, don't apply H to s
        return means, covs, gains
    else:
        if spatiotemporal is False:
            means = np.squeeze(means, axis=-1)
            covs = np.squeeze(covs, axis=-1)
            means = np.transpose(means, axes=[1, 0, 2])
            covs = np.transpose(covs, axes=[1, 0, 2])
        else:
            # return posterior mean and covariance of q(s)
            # means: (num_latent, T, num_space, 1)
            # covs: (num_latent, T, num_space, num_space)
            means = np.transpose(means, axes=[1, 0, 2, 3])
            covs = np.transpose(covs, axes=[1, 0, 2, 3])
            # after transpose
            # means: (T, num_latent, num_space, 1)
            # covs: (T, num_latent, num_space, num_space)
            # gains: (num_latent, T, num_space*state_dim, num_space*state_dim)
        return means, covs, gains


def rauch_tung_striebel_smoother(
    As,
    Pinf,
    H,
    filter_mean,
    filter_cov,
    mask=None,
    block_index=None,
    return_full=False,
    parallel=False,
    spatiotemporal=False,
):
    """
    Run the RTS smoother to get p(fₙ|y₁,...,y_N),
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param filter_mean: the intermediate distribution means computed during filtering [N, state_dim, 1]
    :param filter_cov: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
    :param return_full: a flag determining whether to return the full state distribution or just the function(s)
    :param parallel: flag to switch between parallel and sequential implementation of smoother
    :return:
        smoothed_mean: the posterior marginal means [N, obs_dim]
        smoothed_var: the posterior marginal variances [N, obs_dim]
    """
    # Pinf = kernel.stationary_covariance()
    # As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    # H = kernel.measurement_model()
    if mask is not None:
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
    if mask is None:
        mask = np.zeros(filter_mean.shape[0], dtype=bool)
    else:
        if len(mask.shape) == 4:
            mask = 1 - mask[:, 0, 0, 0]
        elif len(mask.shape) == 3:
            mask = 1 - mask[:, 0, 0]
        else:
            raise ValueError("Mask has incorrect shape")
    if parallel:
        raise NotImplementedError()
        # means, covs, gains = _parallel_rts(
        #     filter_mean, filter_cov, As, Qs, H, return_full
        # )
    else:
        if spatiotemporal == True:
            # filter_mean: (T, num_space, state_dim, 1) 
            # filter_cov: (T, num_space, state_dim, state_dim)
            # As: (T, num_space, state_dim, state_dim)
            # Qs: (T, num_space, state_dim, state_dim)
            # H: (num_space, num_space*state_dim)
            # mask: (T). All are False/0 values if we don't have data mask
            means, covs, gains = _sequential_rts_spatiotemporal(
                filter_mean, filter_cov, As, Qs, H, mask, block_index, return_full
            )
        else:
            means, covs, gains = _sequential_rts(
                filter_mean, filter_cov, As, Qs, H, mask, return_full
            )

    return means, covs, gains


#######################################################################
#################Normal Kalman Filtering and Smoothing#################
#######################################################################
def _sequential_kf(As, Qs, H, ys, noise_covs, m0, P0, masks, return_predict=False):

    def forward_filter(m, P, y, obs_cov, A, Q, ell):
        m_ = A @ m
        P_ = A @ P @ A.T + Q

        obs_mean = H @ m_
        HP = H @ P_
        S = HP @ H.T + obs_cov
        ell_n = mvn_logpdf(y, obs_mean, S)
        ell = ell + ell_n
        K = solve(S, HP).T
        m = m_ + K @ (y - obs_mean)
        P = P_ - K @ HP
        return ell, m, P, m_, P_

    def no_filter(m, P, y, obs_cov, A, Q, ell):
        return ell, m, P, m, P

    def body(carry, inputs):
        y, A, Q, obs_cov, mask = inputs
        m, P, ell = carry
        # TODO: Unit test equality
        ell, m, P, m_, P_ = cond(
            mask[0, 0] == 0, forward_filter, no_filter, m, P, y, obs_cov, A, Q, ell
        )

        if return_predict:
            return (m, P, ell), (m_, P_)
        else:
            return (m, P, ell), (m, P)

    (_, _, loglik), (fms, fPs) = scan(
        f=body, init=(m0, P0, 0.0), xs=(ys, As, Qs, noise_covs, masks)
    )
    return loglik, fms, fPs


def _sequential_rts(fms, fPs, As, Qs, H, masks, return_full):

    state_dim = H.shape[1]

    def backward_smoothing(fm, fP, A, Q, sm, sP):
        pm = A @ fm
        AfP = A @ fP
        pP = AfP @ A.T + Q
        C = solve(pP, AfP).T
        sm = fm + C @ (sm - pm)
        sP = fP + C @ (sP - pP) @ C.T
        return sm, sP, C

    def no_smoothing(fm, fP, A, Q, sm, sP):
        C = np.zeros((state_dim, state_dim))
        return fm, fP, C

    def body(carry, inputs):
        fm, fP, A, Q, mask = inputs
        sm, sP = carry

        # TODO: Unit test equality
        sm, sP, C = cond(
            mask == 0, backward_smoothing, no_smoothing, fm, fP, A, Q, sm, sP
        )

        # pm = A @ fm
        # AfP = A @ fP
        # pP = AfP @ A.T + Q
        # C = solve(pP, AfP).T * (1 - mask)
        # sm = fm + C @ (sm - pm)
        # sP = fP + C @ (sP - pP) @ C.T
        if return_full:
            return (sm, sP), (sm, sP, C)
        else:
            return (sm, sP), (H @ sm, H @ sP @ H.T, C)

    _, (sms, sPs, gains) = scan(
        f=body, init=(fms[-1], fPs[-1]), xs=(fms, fPs, As, Qs, masks), reverse=True
    )
    return sms, sPs, gains


#######################################################################
#############Spatiotemporal Kalman Filtering and Smoothing#############
#######################################################################


def _sequential_kf_spatiotemporal(
    As, Qs, H, ys, noise_covs, m0, P0, masks, block_index
):
    """Kalman filtering loop
    
    As: (T, num_space, state_dim, state_dim)
    Qs: (T, num_space, state_dim, state_dim)
    H: (num_space, num_space*state_dim)
    ys: (T, num_space, 1). This is the variational mean
    noise_covs: (T, num_space, 1). This is the variational variational
    m0: (num_space, state_dim, 1)
    P0: (num_space, state_dim, state_dim)
    masks: (num_space, 1, 1). All are False/0 values if we don't have data mask
    """

    # write down the dimensions
    num_space = m0.shape[0]
    sub_state_dim = m0.shape[1]
    stacked_state_dim = H.shape[1]

    # since we filter over the stacked dimension num_space*state_dim, 
    # we need to create block-diagonal matrices
    Pzeros = np.zeros((stacked_state_dim, stacked_state_dim))
    Pzeros_space = np.zeros((num_space, num_space))

    # this is to extract diagonal values from a [num_space, num_space]
    # matrix
    indices = np.arange(num_space)

    # build a diagonal matrix of size [num_space, num_space] 
    # with this
    def build_diag(values):
        P = Pzeros_space.at[indices, indices].set(values.flatten())
        return P

    # build a block diagonal matrix of size 
    # [num_space*state_dim, num_space*state_dim] 
    # This populates the matrix block-diagonally 
    # according to the block_index tuple containing
    # the x-y coordinates
    def build_block_diag(P_blocks):
        P = Pzeros.at[block_index].set(P_blocks.flatten())
        return P

    # create a stacked mean
    def get_block_mean(m):
        return m.reshape(num_space, sub_state_dim, 1)

    # create a block-diagonal matrix of size [num_space*state_dim, num_space*state_dim] 
    def get_block_cov(P):
        return P.at[block_index].get().reshape(num_space, sub_state_dim, sub_state_dim)

    # helper method to perform lines 4 to 6 in Algorithm 1
    def forward_filter(m, P, y, obs_cov_diag, A, Q, ell):
        m = (A @ m).reshape(-1, 1)
        P = build_block_diag(A @ P @ transpose(A) + Q)

        obs_mean = H @ m
        HP = H @ P
        S = HP @ H.T + obs_cov_diag

        ell_n = mvn_logpdf(y, obs_mean, S)
        ell = ell + ell_n

        K = solve(S, HP).T
        m = get_block_mean(m + K @ (y - obs_mean))
        P = get_block_cov(P - K @ HP)
        return ell, m, P

    # Don't filter and return the previous values
    # This handles irregularly sampled data
    def no_filter(m, P, y, obs_cov_diag, A, Q, ell):
        return ell, m, P

    # main loop 
    # warning: to debug, you can't print anything inside
    # best way is to copy all this code into jupyter notebook
    def body(carry, inputs):
        y, A, Q, obs_cov, mask = inputs
        # Put the variational/encoded variances into
        # a diagonal matrix
        obs_cov_diag = build_diag(obs_cov)
        m, P, ell = carry
        ell, m, P = cond(
            mask[0, 0] == 0, forward_filter, no_filter, m, P, y, obs_cov_diag, A, Q, ell
        )

        return (m, P, ell), (m, P)

    # Scan runs the main loop body()
    # Initial values m0, P0, 0.0
    # it plugs in ys[i], As[i], Qs[i] etc... 
    # each iteration i=1,...,T
    # The values (m, P, ell), (m, P) are updated like in a usual for loop
    (_, _, loglik), (fms, fPs) = scan(
        f=body, init=(m0, P0, 0.0), xs=(ys, As, Qs, noise_covs, masks)
    )
    return loglik, fms, fPs


def _sequential_rts_spatiotemporal(
    fms, fPs, As, Qs, H, masks, block_index, return_full
):
    """
    fms: (T, num_space, state_dim, 1) 
    fPs: (T, num_space, state_dim, state_dim)
    As: (T, num_space, state_dim, state_dim)
    Qs: (T, num_space, state_dim, state_dim)
    H: (num_space, num_space*state_dim)
    masks: (T). All are False/0 values if we don't have data mask
    """

    # write down dimensions
    num_space = fms.shape[1]
    sub_state_dim = fms.shape[2]
    stacked_state_dim = H.shape[1]

    # for creating block diagonal matrices
    # of dimension [num_space*state_dim, num_space*state_dim]
    Pzeros = np.zeros([stacked_state_dim, stacked_state_dim])
    def build_block_diag(P_blocks):
        P = Pzeros.at[block_index].set(P_blocks.flatten())
        return P

    # main method that does like 10 in Algorithm 1
    def backward_smoothing(fm, fP, A, Q, sm, sP):
        pm = A @ fm
        AfP = A @ fP
        pP = AfP @ transpose(A) + Q
        C = transpose(solve(pP, AfP))
        sm = fm + C @ (sm - pm)
        sP = fP + C @ (sP - pP) @ transpose(C)
        return sm, sP, C

    # return previous smoothing values if we don't smooth
    # C=0 so that there's no gain
    def no_smoothing(fm, fP, A, Q, sm, sP):
        C = np.zeros((num_space, sub_state_dim, sub_state_dim))
        return fm, fP, C

    # main loop
    def body(carry, inputs):
        fm, fP, A, Q, mask = inputs
        sm, sP = carry

        # TODO: Unit test equality
        sm, sP, C = cond(
            mask == 0, backward_smoothing, no_smoothing, fm, fP, A, Q, sm, sP
        )

        if return_full:
            return (sm, sP), (
                sm.reshape(-1, 1),
                build_block_diag(sP),
                build_block_diag(C),
            )
        else:
            # double check this bit
            # we're going back to s
            # which is a dimension [num_space*state_dim, 1] vector
            return (sm, sP), (
                H @ sm.reshape(-1, 1),
                H @ build_block_diag(sP) @ transpose(H),
                build_block_diag(C),
            )

    _, (sms, sPs, gains) = scan(
        f=body, init=(fms[-1], fPs[-1]), xs=(fms, fPs, As, Qs, masks), reverse=True
    )
    return sms, sPs, gains
