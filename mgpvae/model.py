from mgpvae.base_model import BaseGPVAEExternalModel
from mgpvae.util import (
    transpose,
    temporal_conditional,
    diag
)
from mgpvae.ops import (
    kalman_filter_independent_latent,
    rauch_tung_striebel_smoother_independent_latent,
)
from mgpvae.util import gaussian_expected_log_lik_diag
from mgpvae.networks import Linear
import jax 
from jax import vmap, random
import jax.numpy as np
from jax.lib import xla_bridge
import objax
from typing import List

#######################################################################
#######################################################################
#####################External##########################################
#######################################################################
#######################################################################
class MarkovGaussianProcessVAEExternal(BaseGPVAEExternalModel):
    """

    Note:
    - The time series has fixed length for each batch
    - The time series is ordered
    
    """
    def __init__(
        self,
        kernel,
        likelihood,
        encoder,
        num_hidden,
        minibatch_size,
        num_sequences,
        dt,
        num_latent=1,
        beta=1,
        hidden_to_mu=None,
        hidden_to_var=None,
        parallel=False,
        missing_mechanism=None,
        time_transform=None,
        cnn=False
    ):
        if parallel is None:  # if using a GPU, then run the parallel filter
            parallel = xla_bridge.get_backend().platform == "gpu"

        self.dt = dt
        self.num_sequences = num_sequences
        self.minibatch_size = minibatch_size
        super().__init__(kernel, likelihood, func_dim=num_latent)
        self.num_latent = num_latent
        self.beta = beta
        self.scale = self.num_sequences / self.minibatch_size

        self.parallel = parallel
        self.encoder = encoder
        self.missing_mechanism=missing_mechanism

        if hidden_to_mu:
            self.hidden_to_mu = hidden_to_mu
        else:
            self.hidden_to_mu = objax.nn.Sequential(
                [
                    Linear(num_hidden * 8 * 8, num_latent),
                ]
            )
        if hidden_to_var:
            self.hidden_to_var = hidden_to_var
        else:
            self.hidden_to_var = objax.nn.Sequential(
                [
                    Linear(num_hidden * 8 * 8, num_latent),
                ]
            )

        self.time_transform = time_transform
        self.cnn = cnn
    @staticmethod
    def filter(dt, kernel, pseudo_y, pseudo_var, mask=None, parallel=False):
        return kalman_filter_independent_latent(
            dt=dt,
            kernel=kernel,
            y=pseudo_y,
            noise_cov=pseudo_var,
            mask=mask,
            parallel=parallel,
        )

    @staticmethod
    def smoother(
        dt, kernel, filter_mean, filter_cov, mask=None,return_full=False, parallel=False
    ):
        return rauch_tung_striebel_smoother_independent_latent(
            dt,
            kernel,
            filter_mean,
            filter_cov,
            mask,
            return_full=return_full,
            parallel=parallel,
        )
    @staticmethod
    def temporal_conditional(*args, **kwargs):
        return temporal_conditional(*args, **kwargs)

    def compute_full_pseudo_lik(self, Y):
        hidden = self.encoder(Y)
        # mean, var = np.split(out, 2, axis=-1)
        # var = objax.functional.softplus(var)

        # mean parameterisation
        # mean = self.hidden_to_mu(hidden)
        # var = self.hidden_to_var(hidden)
        # mean, var = np.expand_dims(mean, axis=-1), np.expand_dims(var, axis=-1)
        # var = objax.functional.softplus(var)

        # natural parameterisation 
        lambda1 = self.hidden_to_mu(hidden)
        lambda2 = self.hidden_to_var(hidden)
        lambda1, lambda2 = np.expand_dims(lambda2, axis=-1), np.expand_dims(lambda2, axis=-1)
        lambda2 = -objax.functional.softplus(lambda2)
        # learn lambda2 <- 1 / lambda2
        mean = lambda1 * lambda2
        var = -0.5 * lambda2

        # sigma = (0.1 + 0.9 * objax.functional.softplus(logsigma))
        # var = objax.functional.softplus(var)*0.05
        # var = objax.functional.softplus(log_std)**2
        # mean, var = np.expand_dims(mean, axis=-1), np.expand_dims(var, axis=-1)
        # mean = np.expand_dims(mean, axis=1)
        # return self.mean.value, objax.functional.softplus(self.var.value)
        return mean, var

    def compute_log_lik(self, pseudo_y, pseudo_var, dt, mask=None):
        """
        int p(f) N(pseudo_y | f, pseudo_var) df
        """
        log_lik_pseudo, (_, _) = self.filter(
            dt,
            self.kernel,
            pseudo_y,
            pseudo_var,
            parallel=self.parallel,
            mask=mask,
        )

        return log_lik_pseudo

    def update_posterior(self, Y, dt=None, mask=None):
        """
        Compute the posterior via filtering and smoothing
        """

        if self.cnn:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y.astype(np.float32))
            pseudo_y, pseudo_var = pseudo_y.astype(np.float64), pseudo_var.astype(np.float64)
        else:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y)
        log_lik_pseudo, (filter_mean, filter_cov) = self.filter(
            dt,
            self.kernel,
            pseudo_y,
            pseudo_var,
            mask=mask,
            parallel=self.parallel,
        )
        dt = np.concatenate([dt[1:], np.array([0.0])], axis=0)
        smoother_mean, smoother_cov, _ = self.smoother(
            dt, self.kernel, filter_mean, filter_cov, parallel=self.parallel, mask=mask, return_full=False
        )
        # self.posterior_mean.value, self.posterior_variance.value = (
        #     smoother_mean,
        #     smoother_cov,
        # )

        return filter_mean, filter_cov, smoother_mean, smoother_cov, pseudo_y, pseudo_var, log_lik_pseudo

    def update_posterior_missing_frame(self, Y, mask, dt=None):
        """
        Compute the posterior via filtering and smoothing
        """
        if self.cnn:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y.astype(np.float32))
            pseudo_y, pseudo_var = pseudo_y.astype(np.float64), pseudo_var.astype(np.float64)
        else:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y)
        log_lik_pseudo, (filter_mean, filter_cov) = self.filter(
            dt,
            self.kernel,
            pseudo_y,
            pseudo_var,
            mask=mask,
            parallel=self.parallel,
        )
        dt = np.concatenate([dt[1:], np.array([0.0])], axis=0)
        smoother_mean, smoother_cov, gain = self.smoother(
            dt, self.kernel, filter_mean, filter_cov, parallel=self.parallel, mask=mask, return_full=False
        )
        # self.posterior_mean.value, self.posterior_variance.value = (
        #     smoother_mean,
        #     smoother_cov,
        # )

        return filter_mean, filter_cov, smoother_mean, smoother_cov, pseudo_y, pseudo_var, log_lik_pseudo
    
    def compute_kl_missing_frame(self, pseudo_y, pseudo_var, posterior_mean, posterior_variance, log_lik_pseudo, mask, train_rng, return_samples=False, num_samples=20):
        """
        KL[q()|p()]
        """
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        # pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y)
        # log_lik_pseudo = self.compute_log_lik(pseudo_y, pseudo_var, dt, mask=mask)
        if return_samples:
            # This is to compute the KL sample-wise KL(q(z_k) || p(z_k))
            # need to take N(0,1) samples from (100, 16, 1)
            # then compute log N(sample_k| pseudo_y, pseudo_var) for k=1,..,K
            # use same train_rng so that we end up with the same random
            # PRNG as in the variational_exp function
            posterior_std = np.sqrt(posterior_variance)
            @vmap
            def generate_normal(posterior_mean, posterior_std, train_rng):
                # split since it is used in the variational expectation function
                # could probably get rid of it in the future
                train_rng = random.split(train_rng, 1)[0]
                eps = random.normal(key=train_rng, shape=(posterior_std.shape[0], num_samples))
                return posterior_mean + posterior_std*eps
            z_samples = generate_normal(posterior_mean, posterior_std, train_rng)
            log_density_pseudo = jax.scipy.stats.norm.logpdf(z_samples, pseudo_y, np.sqrt(pseudo_var))
            mask = mask.reshape(log_density_pseudo.shape[0],-1)[:,0]
            mask = mask.reshape([mask.shape[0], 1, 1])
            kl = np.sum(log_density_pseudo*mask, axis=[0,1]) - log_lik_pseudo
        else:
            expected_density_pseudo = vmap(
                gaussian_expected_log_lik_diag
            )(  # parallel operation
                pseudo_y,
                posterior_mean,
                posterior_variance,
                pseudo_var,
            )
            if len(mask.shape) == 4:
                mask = mask[:, 0]
            kl = (
                np.sum(expected_density_pseudo*mask[:, 0, 0]) - log_lik_pseudo
            )  # KL[approx_post || prior]
        # print(log_lik_pseudo, np.sum(expected_density_pseudo*mask[:, 0, 0]))
        # print(expected_density_pseudo.shape, mask[:,0,0].shape)
        # print(expected_density_pseudo*mask[:, 0, 0])
        return kl

    def compute_kl(self, pseudo_y, pseudo_var, posterior_mean, posterior_variance, log_lik_pseudo, train_rng, return_samples=False, num_samples=20):
        """
        KL[q()|p()]
        """
        # pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y)
        # log_lik_pseudo = self.compute_log_lik(pseudo_y, pseudo_var, dt, mask=mask
        if return_samples:
            # This is to compute the KL sample-wise KL(q(z_k) || p(z_k))
            # need to take N(0,1) samples from (100, 16, 1)
            # then compute log N(sample_k| pseudo_y, pseudo_var) for k=1,..,K
            # use same train_rng so that we end up with the same random
            # PRNG as in the variational_exp function
            posterior_std = np.sqrt(posterior_variance)
            
            @vmap
            def generate_normal(posterior_mean, posterior_std, train_rng):
                # split since it is used in the variational expectation function
                # could probably get rid of it in the future
                train_rng = random.split(train_rng, 1)[0]
                eps = random.normal(key=train_rng, shape=(posterior_std.shape[0], num_samples))
                return posterior_mean + posterior_std*eps

            z_samples = generate_normal(posterior_mean, posterior_std, train_rng)
            log_density_pseudo = jax.scipy.stats.norm.logpdf(z_samples, pseudo_y, np.sqrt(pseudo_var))
            kl = np.sum(log_density_pseudo, axis=[0,1]) - log_lik_pseudo
        else:
            expected_density_pseudo = vmap(
                gaussian_expected_log_lik_diag
            )(  # parallel operation
                pseudo_y,
                posterior_mean,
                posterior_variance,
                pseudo_var,
            )
            kl = (
                np.sum(expected_density_pseudo) - log_lik_pseudo
            )  # KL[approx_post || prior]
        # print(log_lik_pseudo, np.sum(expected_density_pseudo))
        # print(expected_density_pseudo)
        return kl
    def energy_batch(self, Y, train_rng, missing_mask=None, t=None, num_samples=1, return_samples=False, y_true=None):

        flag_missing = 0 if missing_mask is not None else None
        flag_t = 0 if t is not None else None
        flag_ytrue = 0 if y_true is not None else None
        negative_varexp, kl = vmap(self.energy, (0, 0, flag_missing, flag_t,  
        None, None, flag_ytrue))(
            Y, 
            train_rng, 
            missing_mask,
            t,
            num_samples,
            return_samples,
            y_true,
        )
        if return_samples:
            # negative_varexp is actually varexp here
            variational_free_energy = negative_varexp - kl
            return variational_free_energy
        else:
            variational_free_energy = np.mean(negative_varexp + kl) 
            return variational_free_energy, np.mean(negative_varexp), np.mean(kl)
    
    def energy(self, Y, train_rng, missing_mask=None, t=None, num_samples=1, return_samples=False, y_true=None):
        if t is not None:
            # assert len(t.shape) == 2
            dt = np.array([0] + list(np.diff(t, axis=0)[:, 0]))
        else:
            dt = self.dt
        if self.time_transform is not None:
            raise NotImplementedError()
            # t = self.time_transform(t)
            # ind = np.argsort(t[:,0])
            # t = t.at[ind].get()
            # Y = Y.at[ind].get()
        # if missing_frames is not None:
        #     assert missing_mask is None and t is None 
        #     Y = Y[missing_frames == 1]
        #     t = np.linspace(0, missing_frames.shape[0]-1, missing_frames.shape[0])[:, None][missing_frames==1]
        #     dt = np.array([0] + list(np.diff(t, axis=0)[:, 0]))
        # X, Y, _, dt = input_admin(X, Y, None)
        # print("\n\n\n\n\n HIHIHIHIHIHI\n\n\n\n\n\n\n", self.encoder.encoder[0].w.dtype, "\n\n\n\n\n")
        # TODO: WTF encoder dtype changed
        # print(self.encoder.encoder[0].w.dtype)
        if self.missing_mechanism in [None, "corrupt"]:
            _, _, mean_f, cov_f, pseudo_y, pseudo_var, log_lik_pseudo = self.update_posterior(Y, dt=dt)  # perform inference and update variational params
        elif self.missing_mechanism == "missing":
            _, _, mean_f, cov_f, pseudo_y, pseudo_var, log_lik_pseudo = self.update_posterior_missing_frame(Y, mask=missing_mask, dt=dt)  # perform inference and update variational params
        else:
            raise NotImplementedError("missing not implemented")
        # VI expected density is E_q[log p(y|f)]
        train_rng = random.split(train_rng, Y.shape[0])
        if y_true is not None:
            Y = y_true


        if missing_mask is not None:
            expected_density = vmap(
                self.likelihood.variational_expectation, (0, 0, 0, 0, 0, None, None)
            )(Y, mean_f, cov_f, train_rng, missing_mask, num_samples, return_samples)
        else:
            expected_density = vmap(
                self.likelihood.variational_expectation, (0, 0, 0, 0, None, None, None)
            )(Y, mean_f, cov_f, train_rng, None, num_samples, return_samples)
        if self.missing_mechanism in [None, "corrupt"]:
            KL = self.compute_kl(pseudo_y, pseudo_var, mean_f, cov_f, log_lik_pseudo, train_rng, return_samples=return_samples, num_samples=num_samples)  # KL[q(f)|p(f)]
        elif self.missing_mechanism == "missing":
            KL = self.compute_kl_missing_frame(pseudo_y, pseudo_var, mean_f, cov_f, log_lik_pseudo, missing_mask, train_rng, return_samples=return_samples, num_samples=num_samples)  # KL[q(f)|p(f)]
        else:
            raise NotImplementedError("missing not implemented")
        # variational_free_energy = (
        #     -(  # the variational free energy, i.e., the negative ELBO
        #         np.nansum(expected_density)  # nansum accounts for missing data
        #         - KL*self.beta
        #     )
        # )
        if return_samples:
            if self.missing_mechanism == "missing":
                mask = missing_mask.reshape(expected_density.shape[0],-1)[:,0]
                mask = mask.reshape([mask.shape[0], 1, 1])
                # mask the missing frames when computing log_prob
                return np.sum(expected_density*mask, axis=[0,1]), KL
            else:
                return np.sum(expected_density, axis=[0,1]), KL
        else:
            return -np.nansum(expected_density), KL*self.beta

    def compute_nll(self, t_input, y_input, t_test, y_test, mask_input=None, seed=0, num_samples=20):
        if mask_input is not None:
            mask_input = np.array(mask_input)

        if self.missing_mechanism == "missing":
            t = t_input
        else:
            t = None
        if self.missing_mechanism == "missing":
            LL = self.energy_batch(
                y_input, 
                random.split(random.PRNGKey(seed), y_input.shape[0]),
                mask_input,
                t=t,
                num_samples=num_samples,
                return_samples=True,
                y_true=y_input, # only compute NLL on observed frames
            )
        else:
            LL = self.energy_batch(
                y_input, 
                random.split(random.PRNGKey(seed), y_input.shape[0]),
                mask_input,
                t=t,
                num_samples=num_samples,
                return_samples=True,
                y_true=y_test,
            )

        nll = jax.scipy.special.logsumexp(LL, 1) + np.log(1/num_samples)
        if self.missing_mechanism == "missing":
            # compute log E[p(Y|Z)] - i.e. l\log p(Y^c|Y)
            assert mask_input is not None, "Mask has to be present for missing mode"
            # compute RMSE
            _, _, mean_f, var_f = jax.vmap(self.predict_y, (0, None, 0, 0, 0, None))(
                jax.numpy.array(t_input),
                jax.numpy.array(t_test),
                jax.numpy.array(y_input),
                jax.numpy.array(mask_input),
                jax.numpy.arange(y_input.shape[0]),
                num_samples,
            )

            @vmap
            def convert_mask(t_input):
                t_input_ = jax.numpy.array(t_input)[:,0]
                if len(y_input.shape)==5:
                    mask_input_ = jax.numpy.zeros([t_input.shape[0], *y_input.shape[-3:]])
                elif len(y_input.shape)==4:
                    mask_input_ = jax.numpy.zeros([t_input.shape[0], *y_input.shape[-2:]])
                elif len(y_input.shape)==3:
                    mask_input_ = jax.numpy.zeros([t_input.shape[0], y_input.shape[-1]])
                else:
                    raise NotImplementedError("dataset not implemented")
                mask_input_ = mask_input_.at[t_input_.astype(int)].set(1)
                return mask_input_

            mask_input_ = convert_mask(t_input)
            expected_logdensity = vmap(
                self.likelihood.compute_loglikelihood, (0, 0, 0, 0, 0, None, None)
            )(
                y_test, 
                mean_f, 
                var_f, 
                random.split(random.PRNGKey(seed), y_input.shape[0]), 
                1-mask_input_, 
                num_samples,
                True
            )
            expected_logdensity = np.sum(expected_logdensity, axis=1)
            nll2 = jax.scipy.special.logsumexp(expected_logdensity, -1) + np.log(1/num_samples)
            nll += nll2
        return -nll

    def predict(self, X_train, X_test, Y_train, mask_train=None):
        """
        predict at new test locations X
        """
        if len(X_test.shape) < 2:
            X_test = X_test[:, None]
        if len(X_train.shape) < 2:
            X_train = X_train[:, None]
        if self.time_transform is not None:
            raise NotImplementedError()
            # X_train = self.time_transform(X_train)
            # ind = np.argsort(X_train[:,0])
            # X_train = X_train.at[ind].get()
            # Y_train = Y_train.at[ind].get()

            # X = self.time_transform(X)
            # ind = np.argsort(X[:,0])
            # X = X.at[ind].get()
        dt = np.concatenate([np.array([0.0]), np.diff(X_train[:, 0])])
        if self.cnn:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y_train.astype(np.float32))
            pseudo_y, pseudo_var = pseudo_y.astype(np.float64), pseudo_var.astype(np.float64)
        else:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y_train)
        _, (filter_mean, filter_cov) = self.filter(
            dt,
            self.kernel,
            pseudo_y,
            pseudo_var,
            mask=mask_train,
            parallel=self.parallel
        )
        dt = np.concatenate([dt[1:], np.array([0.0])], axis=0)
        smoother_mean, smoother_cov, gain = self.smoother(
            dt,
            self.kernel,
            filter_mean,
            filter_cov,
            return_full=True,
            mask=mask_train,
            parallel=self.parallel,
        )

        # add dummy states at either edge
        inf = 1e10 * np.ones_like(X_train[0, :1])
        X_aug = np.block([[-inf], [X_train[:, :1]], [inf]])
        # smoother_cov = make_diag(smoother_cov)

        # predict the state distribution at the test time steps:
        ind_test = (
            np.searchsorted(
                X_aug.reshape(
                    -1,
                ),
                X_test.reshape(
                    -1,
                ),
            )
            - 1
        )

        def compute_transitions(X_test, ind, X_aug):
            dt_fwd = X_test[..., 0] - X_aug[ind, 0]
            dt_back = X_aug[ind + 1, 0] - X_test[..., 0]
            A_fwd = self.kernel.state_transition(dt_fwd)
            A_back = self.kernel.state_transition(dt_back)
            return A_fwd, A_back 
        A_fwd, A_back = vmap(compute_transitions, (0, 0, None))(
            X_test, ind_test, X_aug
        )
        Pinf = self.kernel.stationary_covariance()
        # print(A_fwd.shape, A_back.shape, Pinf.shape, gain.shape)
        state_mean, state_cov = vmap(self.temporal_conditional, (None, None, 0, 0, 0, 1, 1, 0))(
            X_aug, X_test, smoother_mean, smoother_cov, gain, A_fwd, A_back, Pinf
        )
        # extract function values from the state:
        H = self.kernel.measurement_model()
        def compute_test(state_mean, state_cov, H):
            test_mean, test_var = H @ state_mean, H @ state_cov @ transpose(H)
            return test_mean, test_var
        # print(H.shape, state_mean.shape, state_cov.shape)
        test_mean, test_var = vmap(compute_test, (0, 0, None))(state_mean, state_cov, H[0])
        # train_mean, train_var = vmap(compute_test, (0, 0, None))(smoother_mean, smoother_cov, H[0])

        return np.squeeze(test_mean).T, np.squeeze(test_var).T
        # return np.squeeze(test_mean).T, np.squeeze(test_var).T, np.squeeze(train_mean).T, np.squeeze(train_var).T, smoother_mean, smoother_cov, gain

###########################################################################################################################################################################################################################################################################################################################

## Spatiotemporal GP
class STMarkovGaussianProcessVAEExternal(BaseGPVAEExternalModel):
    """

    Note:
    - The time series has fixed length for each batch
    - The time series is ordered
    
    """
    def __init__(
        self,
        kernel,
        likelihood,
        encoder,
        num_hidden,
        minibatch_size,
        num_sequences,
        dt,
        num_latent=1,
        beta=1,
        hidden_to_mu=None,
        hidden_to_var=None,
        parallel=False,
        missing_mechanism=None,
        time_transform=None
    ):
        if parallel is None:  # if using a GPU, then run the parallel filter
            parallel = xla_bridge.get_backend().platform == "gpu"
        self.parallel = parallel
        
        # NOT USED
        self.dt = dt
        self.num_sequences = num_sequences
        self.minibatch_size = minibatch_size
        self.scale = self.num_sequences / self.minibatch_size
        self.missing_mechanism = missing_mechanism
        self.time_transform = time_transform
        super().__init__(kernel, likelihood, func_dim=num_latent)

        # USED
        self.num_latent = num_latent
        self.beta = beta
        self.encoder = encoder

        # last layer of the encoder
        if hidden_to_mu:
            self.hidden_to_mu = hidden_to_mu
        else:
            self.hidden_to_mu = objax.nn.Sequential(
                [
                    Linear(num_hidden * 8 * 8, num_latent),
                ]
            )
        if hidden_to_var:
            self.hidden_to_var = hidden_to_var
        else:
            self.hidden_to_var = objax.nn.Sequential(
                [
                    Linear(num_hidden * 8 * 8, num_latent),
                ]
            )

    @staticmethod
    def filter(dt, kernel, pseudo_y, pseudo_var, mask=None, parallel=False):
        """Perform Kalman filtering (see ops.py)

        kalman_filter_independent_latent -> kalman_filter() -> _sequential_kf_spatiotemporal()

        Inputs:
            - kernel. This helps output the transition matrix A, stationary covariance matrix 
            Q and emission matrix H
            - pseudo_y: \tilde{y}, the encoded means
            - pseudo_var: \tilde{V}, the encoded variances
            - mask: missing frames mask
            - parallel: use parallel filtering or not (Not implemented yet)

        Returns:
            - E3 (the log-marginal of the encoded variational distribution)
            - 1 and 2: (filter_mean, filter_covs)
        """
        return kalman_filter_independent_latent(
            dt=dt,
            kernel=kernel,
            y=pseudo_y,
            noise_cov=pseudo_var,
            mask=mask,
            parallel=parallel,
            spatiotemporal=True,
        )

    @staticmethod
    def smoother(
        dt, kernel, filter_mean, filter_cov, mask=None,return_full=False, parallel=False
    ):
        """Performs Kalman smoothing (backward solve). Gives q(s_t) for t=1,...,T

        rauch_tung_striebel_smoother_independent_latent() -> rauch_tung_striebel_smoother()
        -> _sequential_rts_spatiotemporal()

        Inputs:
            - The step size dt when performing backward smoothing
            - kernel. This helps output the transition matrix A, stationary covariance matrix 
            Q and emission matrix H
            - filter_mean: 1, the filter distribution mean
            - filter_cov: 2, the filter distribution cov
            - mask: missing frames mask
            - return_full: return full smoothing cov
            - parallel: use parallel smoothing or not (Not implemented yet)

        Returns:
            - means: posterior state mean if return_full or posterior latent process mean
            - covs: posterior cov mean if return_full or posterior latent process cov
            - gains: Kalman gain

        """
        return rauch_tung_striebel_smoother_independent_latent(
            dt,
            kernel,
            filter_mean,
            filter_cov,
            mask,
            return_full=return_full,
            parallel=parallel,
            spatiotemporal=True
        )
    @staticmethod
    def temporal_conditional(*args, **kwargs):
        """Computes the posterior distribution of q(s*) given q(s1)...q(sT).
        Based on Adam et al. (2020)
        """
        return temporal_conditional(*args, **kwargs)

    def compute_full_pseudo_lik(self, Y):
        """Apply encoder to observations. 
        Inputs:
            - Y: high-dimensional data of shape (num_space, num_time, y_dim)
        """

        # Apply encoder to each spatial coordinate
        mean, var = vmap(self._compute_full_pseudo_lik, 0)(Y)
        # Move the spatial dimension to the end
        mean = np.transpose(mean, [1,2,0])
        var = np.transpose(var, [1,2,0])
        return mean, var

    def _compute_full_pseudo_lik(self, Y):
        """Helper function to apply the encoder to Y.
        Inputs:
            - Y: high-dimensional data of shape (num_time, y_dim)
        """

        # Compute hidden representations
        hidden = self.encoder(Y)
        # natural parameterisation 
        lambda1 = self.hidden_to_mu(hidden)
        lambda2 = self.hidden_to_var(hidden)
        lambda2 = -objax.functional.softplus(lambda2)
        # learn lambda2 <- 1 / lambda2
        mean = lambda1 * lambda2
        var = -0.5 * lambda2
        return mean, var

    def update_posterior(self, Y, dt, mask=None):
        """
        Compute the posterior states via filtering and smoothing

        Inputs:
            - Y: high-dimensional data of shape (num_space, num_time, y_dim)
            - dt: the step sizes between observations
            - mask: missing frames mask

        Returns:
            - Filter mean
            - Filter covariance
            - Posterior/smoothing mean
            - Posterior/smoothing covariance
            - Encoded means
            - Encoded variance
            - log-marginal likelihood of encoded variational distribution: E3 
        """

        # pass the data into the encoder
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y)
        
        # Compute the cholesky decomposition of the covariance matrix Ktt' = k(t, t')
        self.kernel.precompute_spatial_mixing()

        # Perform kalman filtering
        log_lik_pseudo, (filter_mean, filter_cov) = self.filter(
            dt,
            self.kernel,
            pseudo_y,
            pseudo_var,
            mask=mask,
            parallel=self.parallel,
        )
        # Since we need to do backward smoothing, we shift the dt to the right and 
        # pad a 0.0 to the end, so that the final datapoint is the start
        # Then perform kalman smoothing
        # return_full=False because we only need the posterior mean
        # i.e. z = Hs
        dt = np.concatenate([dt[1:], np.array([0.0])], axis=0)
        smoother_mean, smoother_cov, _ = self.smoother(
            dt, self.kernel, filter_mean, filter_cov, parallel=self.parallel, mask=mask, return_full=False
        )

        return filter_mean, filter_cov, smoother_mean, smoother_cov, pseudo_y, pseudo_var, log_lik_pseudo

    def compute_kl(self, pseudo_y, pseudo_var, posterior_mean, posterior_variance, log_lik_pseudo):
        """This computes E1 - E3, which is KL[q()|p()]

        Inputs:
            - Encoded means
            - Encoded variance
            - Posterior mean
            - Posterior variance
            - log-marginal likelihood of encoded variational distribution: E3

        Returns:
            - E1 - E3
        """

        # Compute E1
        # This is the expectation of a log-Gaussian over a Gaussian
        # The inside of the expectation is a product of independent Gaussian
        # i.e. mean field variational approximation
        # This gives a shape of T, because we are basically summing up 
        # this expectation over space too (since \tilde{V} is diagonal)
        expected_density_pseudo = vmap(
            gaussian_expected_log_lik_diag
        )(  # parallel operation
            pseudo_y,
            np.squeeze(posterior_mean),
            np.squeeze(posterior_variance),
            pseudo_var,
        )

        # Now sum up the KL divergence
        kl = (
            np.sum(expected_density_pseudo) - log_lik_pseudo
        )  # KL[approx_post || prior]
        return kl

    def energy(self, Y, train_rng, t=None, num_samples=1):

        # If t is given, compute the step size dt between each observation point
        if t is not None:
            # assert len(t.shape) == 2
            dt = np.array([0] + list(np.diff(t, axis=0)[:, 0]))
        else:
            # not really used
            dt = self.dt
        if self.time_transform is not None:
            raise NotImplementedError()
        
        # obtain the posterior mean and covariance
        # Shape size 
        # mean_f: (T, L, num_space, 1)
        # cov_f: (T, L, num_space, num_space)
        # pseudo_y: (T, L, num_space)
        # pseudo_var: (T, L, num_space)
        _, _, mean_f, cov_f, pseudo_y, pseudo_var, log_lik_pseudo = self.update_posterior(Y, dt=dt)  
        cov_f_mf = cov_f.at[:,:,np.arange(self.kernel.Ns), np.arange(self.kernel.Ns)].get()
        cov_f_mf = np.expand_dims(cov_f_mf, -1)
        KL = self.compute_kl(pseudo_y, pseudo_var, mean_f, cov_f_mf, log_lik_pseudo)  # KL[q(f)|p(f)]


        # perform inference and update variational params
        # VI expected density is E_q[log p(y|f)]
        # return cov_f
        # train_rng = random.split(train_rng, Y.shape[1])
        # expected_density = vmap(
        #     self.likelihood.variational_expectation_full, (1, 0, 0, 0, None, None)
        # )(Y, mean_f, cov_f, train_rng, None, num_samples)

        # split the training key. This is required because for each random data generation
        # JAX requires you to input the random key value
        # split it along the spatial axis
        train_rng = random.split(train_rng, Y.shape[0])
        # Now add in spatial mixing to compute the likelihood
        mean_f = np.einsum("ij, tljk -> tlik", self.kernel.Lss, mean_f)
        cov_f = np.einsum("ij, tljk -> tlik", self.kernel.Lss, cov_f)
        # Here I am extracting the diagonal entries of the posterior covariance
        # https://github.com/AaltoML/BayesNewton/blob/cd63b9be625244bf7212ebb59fbf63a69038dbf6/bayesnewton/likelihoods.py#L372
        # Done here as well. This is because we are using the mean field assumption
        cov_f = cov_f.at[:,:,np.arange(self.kernel.Ns), np.arange(self.kernel.Ns)].get()
        cov_f = np.expand_dims(cov_f, -1)
        
        # compute the KL divergence
        

        # Compute E2, the reconstruction error
        # Shape size 
        # Y: (num_space, T, y_dim)
        # mean_f: (T, L, num_space, 1)
        # cov_f: (T, L, num_space, 1)
        # train_rng: (num_space, 2). The 2 is just how JAX stores the key
        expected_density = vmap(
            self.expected_density, (0, 2, 2, 0, None, None)
        )(Y, mean_f, cov_f, train_rng, None, num_samples)
        
        
        # variational_free_energy = (
        #     -(  # the variational free energy, i.e., the negative ELBO
        #         np.nansum(expected_density)  # nansum accounts for missing data
        #         - KL*self.beta
        #     )
        # )
        
        # multiply KL with beta
        kl = KL*self.beta
        # take -E2
        negative_varexp = -np.nansum(expected_density) 

        # first output is loss = - E2 + beta*(E1  - E3)
        # second output is -E2
        # third output is beta*(E1 - E3)
        return negative_varexp + kl, negative_varexp, kl

    def expected_density(self, Y, mean_f, cov_f, train_rng, missing_mask, num_samples):
        """Helper function that helps compute batch-wise variational expectation/reconstruction
        # Shape size 
        # Y: (T, y_dim)
        # mean_f: (T, L, 1)
        # cov_f: (T, L, 1)
        # train_rng: (,2). The 2 is just how JAX stores the key
        """
        # Now split the seed again on the time axis
        train_rng = random.split(train_rng, mean_f.shape[0])
        if missing_mask is not None:
            expected_density = vmap(
                self.likelihood.variational_expectation, (0, 0, 0, 0, 0, None)
            )(Y, mean_f, cov_f, train_rng, missing_mask, num_samples)
        else:
            expected_density = vmap(
                self.likelihood.variational_expectation, (0, 0, 0, 0, None, None)
            )(Y, mean_f, cov_f, train_rng, None, num_samples)
        return expected_density

    def compute_nll(self, t_input, y_input, t_test, y_test, R_test, seed=0, num_samples=20):
        """Compute the negative loglikelihood
        """
        mean_y, _, mean_f, var_f = self.predict_y(
            np.array(t_input),
            np.array(t_test),
            np.array(y_input),
            np.array(R_test),
            seed=seed,
            num_samples=num_samples,
        )
        train_rng = random.split(random.PRNGKey(seed), y_test.shape[0])
        expected_density = vmap(
            self.likelihood.compute_loglikelihood, (0, 0, 0, 0, None, None, None, None)
        )(y_test, mean_f, var_f, train_rng, None, num_samples, True, True)
        expected_density = np.sum(expected_density, 3) 
        nll = jax.scipy.special.logsumexp(expected_density, 2)+np.log(1/num_samples)
        nll = np.mean(nll, [0, 1])
        return -nll, expected_density, mean_f, var_f

    def predict(self, X_train, X_test, Y_train, R_test):
        """
        predict at new test locations (X_*, R_*), where X_* is a new time location

        spatial_dim = number of dimensions of the spatial variable 
        e.g. (latitude, longitude, elevation)
        
        Inputs:
            - X_train: (T)
            - X_test: (T)
            - Y_train: (num_space, T, y_dim)
            - R_test: (num_space, spatial_dim)
        
        Returns:
            - Posterior GP mean new test locations
            - Posterior GP variance new test locations
        """


        if len(X_test.shape) < 2:
            X_test = X_test[:, None]
        if len(X_train.shape) < 2:
            X_train = X_train[:, None]
        if self.time_transform is not None:
            raise NotImplementedError()
        assert len(X_test.shape) == 2
        dt = np.concatenate([np.array([0.0]), np.diff(X_train[:, 0])])
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y_train)
        self.kernel.precompute_spatial_mixing()
        _, (filter_mean, filter_cov) = self.filter(
            dt,
            self.kernel,
            pseudo_y,
            pseudo_var,
            parallel=self.parallel
        )
        dt = np.concatenate([dt[1:], np.array([0.0])], axis=0)
        smoother_mean, smoother_cov, gain = self.smoother(
            dt,
            self.kernel,
            filter_mean,
            filter_cov,
            return_full=True,
            parallel=self.parallel,
        )

        # add dummy states at either edge
        inf = 1e10 * np.ones_like(X_train[0, :1])
        X_aug = np.block([[-inf], [X_train[:, :1]], [inf]])
        # smoother_cov = make_diag(smoother_cov)

        # predict the state distribution at the test time steps:
        ind_test = (
            np.searchsorted(
                X_aug.reshape(
                    -1,
                ),
                X_test.reshape(
                    -1,
                ),
            )
            - 1
        )

        def compute_transitions(X_test, ind, X_aug):
            dt_fwd = X_test[..., 0] - X_aug[ind, 0]
            dt_back = X_aug[ind + 1, 0] - X_test[..., 0]
            A_fwd = self.kernel.state_transition_stacked(dt_fwd)
            A_back = self.kernel.state_transition_stacked(dt_back)
            return A_fwd, A_back 
        A_fwd, A_back = vmap(compute_transitions, (0, 0, None))(
            X_test, ind_test, X_aug
        )
        Pinf = self.kernel.stationary_covariance_stacked()
        # print(A_fwd.shape, A_back.shape, Pinf.shape, gain.shape)
        state_mean, state_cov = vmap(self.temporal_conditional, (None, None, 0, 0, 0, 1, 1, 0))(
            X_aug, X_test, smoother_mean, smoother_cov, gain, A_fwd, A_back, Pinf
        )

        # extract function values from the state:
        H = self.kernel.measurement_model()

        # This is to include the spatial mixing matrix
        # H = self.kernel.Lss @ H

        def compute_test(state_mean, state_cov, H):
            test_mean, test_var = H @ state_mean, H @ state_cov @ transpose(H)
            return test_mean, diag(test_var)
        
        # just to vmap over both latent dimensions and time
        def compute_emission(W, state_mean, state_cov, C):
            return vmap(compute_emission_, (None, 0, 0, 0))(W, state_mean, state_cov, C)
        def compute_emission_(W, state_mean, state_cov, C):
            test_mean = W @ state_mean
            test_var = W @ state_cov @ transpose(W) + C
            return test_mean, np.diagonal(test_var)
        def compute_spatial_test(state_mean, state_cov, H):
            # see equation 22 in 
            # https://proceedings.mlr.press/v130/wilkinson21a/wilkinson21a.pdf
            # TODO: if R is fixed, only compute B, C once
            B, C = self.kernel.spatial_conditional(X_test, R_test)
            W = np.einsum("ij,ljk->lik", B, H)
            return vmap(compute_emission, (0, 0, 0, 1))(W, state_mean, state_cov, C)

        if R_test is None:
            test_mean, test_var = vmap(compute_test, (0, 0, 0))(state_mean, state_cov, H)
        else:
            test_mean, test_var = compute_spatial_test(state_mean, state_cov, H)
        return np.squeeze(test_mean).T, np.squeeze(test_var).T
    
    def predict_y(self, X_train, X_test, Y_train, R_test=None, seed=0, num_samples=1):
        """
        predict y at new test locations X
        """
        # mean_f: num_space x T x num_latent
        mean_f, var_f = self.predict(X_train, X_test, Y_train, R_test)
        # mean_f: num_space x T x num_latent x 1
        mean_f = np.expand_dims(mean_f, -1)
        var_f = np.expand_dims(var_f, -1)
        pred_rng = random.PRNGKey(seed)
        pred_rng = random.split(pred_rng, mean_f.shape[0])

        mean_y, var_y = vmap(self.likelihood_predict, (0, 0, 0, None))(
            mean_f, var_f, pred_rng, num_samples
        )
        return np.squeeze(mean_y), np.squeeze(var_y), mean_f, var_f

    def likelihood_predict(self, mean_f, var_f, pred_rng, num_samples):
        pred_rng = random.split(pred_rng, mean_f.shape[0])
        mean_y, var_y = vmap(self.likelihood.predict, (0, 0, 0, None))(
            mean_f, var_f, pred_rng, num_samples
        )
        return mean_y, var_y