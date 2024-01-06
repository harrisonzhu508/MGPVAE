from urllib.request import proxy_bypass
import jax.numpy as np
from jax import vmap
from jax import random
import objax
from mgpvae.util import softplus, softplus_inv
from abc import abstractmethod
import jax 
from jax.scipy.special import expit
class Likelihood(objax.Module):
    """Abstract Likelihood Class

    """

    def __init__(self, decoder, y_dim=1, num_latent=None, cnn=False):
        super().__init__()
        self.decoder = decoder  # outputs the mean
        self.y_dim = y_dim
        if num_latent is None:
            self.num_latent = list(decoder.vars())[1].shape[0]
        else:
            self.num_latent = num_latent
        self.cnn = cnn

    @abstractmethod
    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|φ(fₙ),σ²), where φ is the decoder network
        Can be used to evaluate Q Monte Carlo points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|φ(fₙ),σ²), where σ² is the observation noise variance [Q, 1]
        """

        raise NotImplementedError()

    @abstractmethod
    def conditional_moments(self, f):
        """
        """
        raise NotImplementedError()

    def variational_expectation(self, y, m, v, train_rng, missing_mask, num_samples=1, return_samples=False):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """
        # align shapes and compute mask
        # 1 channel
        if isinstance(self.y_dim, tuple):
            y = y.reshape(-1, 1, *self.y_dim)
            if missing_mask is not None:
                missing_mask = missing_mask.reshape(-1, 1, *self.y_dim)
        else:
            y = y.reshape(-1, 1, self.y_dim)
            if missing_mask is not None:
                missing_mask = missing_mask.reshape(-1, 1, self.y_dim)
        m = m.reshape(-1, self.num_latent, 1)
        # v is a vector of variances already, not a diagonal matrix
        # v = np.diag(v).reshape(-1, self.num_latent, 1)
        v = v.reshape(-1, self.num_latent, 1)
        # mask = np.isnan(y)
        # y = np.where(mask, m, y)
        # compute variational expectations and their derivatives
        train_rng = random.split(train_rng, y.shape[0])
        if missing_mask == None:
            var_exp = vmap(self.variational_expectation_, (0, 0, 0, 0, None, None, None))(y, m, v, train_rng, missing_mask, num_samples, return_samples)
        else:
            var_exp = vmap(self.variational_expectation_, (0, 0, 0, 0, 0, None, None))(y, m, v, train_rng, missing_mask, num_samples, return_samples)
        return var_exp

    def compute_loglikelihood(self, y, m, v, train_rng, missing_mask, num_samples=1, return_samples=False, spatiotemporal=False):
        # align shapes and compute mask
        # 1 channel
        if isinstance(self.y_dim, tuple):
            y = y.reshape(-1, 1, *self.y_dim)
            if missing_mask is not None:
                missing_mask = missing_mask.reshape(-1, 1, *self.y_dim)
        else:
            y = y.reshape(-1, 1, self.y_dim)
            if missing_mask is not None:
                missing_mask = missing_mask.reshape(-1, 1, self.y_dim)
        m = m.reshape(-1, self.num_latent, 1)
        # v is a vector of variances already, not a diagonal matrix
        # v = np.diag(v).reshape(-1, self.num_latent, 1)
        v = v.reshape(-1, self.num_latent, 1)
        # mask = np.isnan(y)
        # y = np.where(mask, m, y)
        # compute variational expectations and their derivatives
        train_rng = random.split(train_rng, y.shape[0])
        var_exp = vmap(self.test_loglikelihood_, (0, 0, 0, 0, 0, None, None, None))(y, m, v, train_rng, missing_mask, num_samples, return_samples, spatiotemporal)

        # apply mask
        return var_exp

    def test_loglikelihood_(self, y, m, v, train_rng, missing_mask, num_samples=1, return_samples=False, spatiotemporal=False):
        """
        If no custom variational expectation method is provided, we use Monte Carlo.
        """
        return test_loglikelihood_monte_carlo(self, y, m, v, train_rng, missing_mask, num_samples=num_samples, return_samples=return_samples, spatiotemporal=spatiotemporal)

    def variational_expectation_(self, y, m, v, train_rng, missing_mask, num_samples=1, return_samples=False):
        """
        If no custom variational expectation method is provided, we use Monte Carlo.
        """
        return variational_expectation_monte_carlo(self, y, m, v, train_rng, missing_mask, num_samples=num_samples, return_samples=return_samples)

    def predict(self, mean_f, var_f, pred_rng, num_samples=1):
        """
        predict in data space given predictive mean and var of the latent function
        """
        return predict_montecarlo(self, mean_f, var_f, pred_rng=pred_rng, num_samples=num_samples)


class DecoderGaussian(Likelihood):
    """The Gaussian likelihood with a decoder network mapping the latent function
    to the mean

    """

    def __init__(self, decoder, y_dim=1, variance=0.1, fix_variance=False, num_latent=None, cnn=False):
        super().__init__(decoder, y_dim, num_latent, cnn)
        if fix_variance:
            self.transformed_variance = objax.StateVar(np.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(np.array(softplus_inv(variance)))
        self.name = 'DecoderGaussian'
        self.link_fn = lambda f: f
        

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def variational_expectation_full(self, y, m, cov, train_rng, missing_mask, num_samples=1):
        post_chol = np.linalg.cholesky(cov)
        m = m[:,:,0] # remove redundant dimension
        sigma_points = np.einsum("lij, blj -> bli", cov, random.normal(key=train_rng, shape=(num_samples, post_chol.shape[0], post_chol.shape[1]))) + m
        sigma_points = np.transpose(sigma_points, [0,2,1])

        sigma_points = sigma_points.astype(np.float32)
        mean = self.decoder(sigma_points).astype(np.float64)
        exp_log_lik = -0.5 * np.log(2 * np.pi * self.variance) - 0.5 * (y[None] - mean) ** 2 / self.variance

        exp_log_lik = np.mean(exp_log_lik, axis=0)
        if missing_mask is not None:
            missing_mask = np.squeeze(missing_mask)
            exp_log_lik = np.where(missing_mask, exp_log_lik, 0)
        if len(exp_log_lik.shape) > 0:
            exp_log_lik = np.sum(exp_log_lik)
        return exp_log_lik

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|φ(fₙ),σ²), where φ is the decoder network
        Can be used to evaluate Q Monte Carlo points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|φ(fₙ),σ²), where σ² is the observation noise variance [Q, 1]
        """
        # f here is of dimension 1 x num_data
        if self.cnn:
            f = f.astype(np.float32)
        if f.shape[0] == self.num_latent:
            mean = self.decoder(f.T)
        else:
            mean = self.decoder(f)
        if self.cnn:
            mean = mean.astype(np.float64)
        out = -0.5 * np.log(2 * np.pi * self.variance) - 0.5 * (y - mean) ** 2 / self.variance
        if len(out.shape) == 2:
            return out
        elif out.shape[1] == 3:
            return out
        else:
            return np.squeeze(out, axis=1)

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Gaussian are the mean and variance:
            E[y|f] = f
            Var[y|f] = σ²
        """
        # f here is of dimension 1 x num_data
        if self.cnn:
            f = f.astype(np.float32)
            
        if f.shape[0] == self.num_latent:
            mean = self.decoder(f.T)
        else:
            mean = self.decoder(f)
        if self.cnn:
            mean = mean.astype(np.float64)
        return mean, np.array([[self.variance]])
        
    def evaluate_likelihood(self, y, f):
        """
        :param y: observed data yₙ [scalar]
        :param f: latent function value fₙ ϵ ℝ^L
        :return:
            p(yₙ|fₙ) 
        """
        # f here is of dimension 1 x num_data
        f = f.astype(np.float32)
        if f.shape[0] == self.num_latent:
            mean = self.decoder(f.T).astype(np.float64)
        else:
            mean = self.decoder(f).astype(np.float64)
        return jax.scipy.stats.norm.pdf(y, loc=mean, scale=np.sqrt(self.variance))
class DecoderBernoulli(Likelihood):
    """The Bernoulli likelihood with a decoder network mapping the latent function
    to the mean

    """

    def __init__(self, decoder, y_dim=1, num_latent=None, cnn=False):
        super().__init__(decoder, y_dim, num_latent, cnn)
        self.name = 'DecoderBernoulli'
        self.link_fn = expit

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the log-Bernoulli function log p(yₙ|φ(fₙ)), where φ is the decoder network
        Can be used to evaluate Q Monte Carlo points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log p(yₙ|φ(fₙ)), where σ² is the observation noise variance [Q, 1]
        """
        # f here is of dimension num_latent x num_data
        if self.cnn:
            f = f.astype(np.float32)
        if f.shape[0] == self.num_latent:
            mean = self.decoder(f.T)
        else:
            mean = self.decoder(f)
        if self.cnn:
            mean = mean.astype(np.float64)
        probs = self.link_fn(mean)
        out = np.where(np.equal(y, 1), probs, 1 - probs)
        return np.log(np.squeeze(out, axis=1))

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Gaussian are the mean and variance:
            E[y|f] = f
            Var[y|f] = σ²
        """
        # f here is of dimension 1 x num_data
        f = f.astype(np.float32)
        if f.shape[0] == self.num_latent:
            mean = self.decoder(f.T).astype(np.float64)
        else:
            mean = self.decoder(f).astype(np.float64)
        return self.link_fn(mean), self.link_fn(mean)-(self.link_fn(mean)**2)

    def evaluate_likelihood(self, y, f):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :return:
            p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾
        """
        # f here is of dimension 1 x num_data
        if self.cnn:
            f = f.astype(np.float32)
        if f.shape[0] == self.num_latent:
            mean = self.decoder(f.T)
        else:
            mean = self.decoder(f)
        if self.cnn:
            mean = mean.astype(np.float64)
        return np.where(np.equal(y, 1), self.link_fn(mean), 1 - self.link_fn(mean))


def predict_montecarlo(likelihood, mean_f, var_f, pred_rng, num_samples=1):
    """
    predict in data space given predictive mean and var of the latent function
    """
    # num_latent x num_latent
    # var_f = (var_f + var_f.T) / 2
    # print(var_f.shape)
    # chol_f, low = cho_factor(var_f, lower=True)
    # if len(var_f.shape) > 0:
    #     chol_f = np.sqrt(np.diag(var_f))[:, None]
    #     num_latent = chol_f.shape[0]
    # else:
        # chol_f = np.sqrt(var_f)
    #     num_latent = 1
    # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to latent dist.
    num_latent = var_f.shape[0]
    chol_f = np.sqrt(var_f)
    # num_latent x num_latent
    sigma_points = chol_f * random.normal(key=pred_rng, shape=(num_latent, num_samples)) + mean_f
    # Compute moments via Monte Carlo:
    # E[y] = ∫ E[yₙ|fₙ] N(fₙ|mₙ,vₙ) dfₙ
    #      ≈ ∑ᵢ wᵢ E[yₙ|fₙ]
    # E[y^2] = ∫ (Cov[yₙ|fₙ] + E[yₙ|fₙ]^2) N(fₙ|mₙ,vₙ) dfₙ
    #        ≈ ∑ᵢ wᵢ (Cov[yₙ|fₙ] + E[yₙ|fₙ]^2)

    # conditional_expectation: (20, D_y)
    # conditional_covariance: (noise_dim, noise_dim)
    conditional_expectation, conditional_covariance = likelihood.conditional_moments(sigma_points)
    # conditional_expectation: (D_y, 20)
    # conditional_expectation = np.squeeze(conditional_expectation)
    # conditional_expectation = conditional_expectation.T
    # expected_y: (D_y,)
    expected_y = conditional_expectation
    # conditional_expectation_squared: 
    conditional_expectation_squared = conditional_expectation ** 2
    # expected_y_squared: (20, 1) x [(1, 1) + (1, D_y, 20)] -> (1, 1)
    if len(conditional_expectation_squared.shape) < 2:
        conditional_expectation_squared = np.expand_dims(conditional_expectation_squared, axis=-1)
    expected_y_squared = np.mean(
        conditional_covariance + conditional_expectation_squared,
        axis=0
    )
    # covariance_y = expected_y_squared - np.mean(expected_y ** 2, axis=0)
    covariance_y = expected_y_squared - np.mean(expected_y, axis=0) ** 2 
    return np.mean(expected_y, axis=0), covariance_y

    # conditional_expectation, conditional_covariance = likelihood.conditional_moments(sigma_points)
    # expected_y = np.sum(w * conditional_expectation, axis=-1)
    # conditional_expectation_ = conditional_expectation.T[..., None]
    # conditional_expectation_squared = conditional_expectation_ @ transpose(conditional_expectation_)
    # expected_y_squared = np.sum(
    #     w * (conditional_covariance + conditional_expectation_squared.T),
    #     axis=-1
    # )
    # # Cov[y] = E[y^2] - E[y]^2
    # covariance_y = expected_y_squared - expected_y[..., None] @ expected_y[None]
    # return expected_y, covariance_y

def variational_expectation_monte_carlo(likelihood, y, post_mean, post_cov, train_rng, missing_mask, num_samples=1, return_samples=False):
    """
    Computes the "variational expectation" via Monte Carlo, i.e. the
    expected log-likelihood, and its derivatives w.r.t. the posterior mean
        E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) N(fₙ|mₙ,vₙ) dfₙ
    with EP power a.
    :param likelihood: the likelihood model
    :param y: observed data (yₙ) [scalar]
    :param post_mean: posterior mean (mₙ) [scalar]
    :param post_cov: posterior variance (vₙ) [scalar]
    :param Monte Carlo: the function to compute sigma points and weights to use during Monte Carlo
    :return:
        exp_log_lik: the expected log likelihood, E[log p(yₙ|fₙ)]  [scalar]
        dE_dm: derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
        d2E_dm2: second derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
    """
    # num_latent x num_latent
    # post_cov = (post_cov + post_cov.T) / 2
    # chol_f, low = cho_factor(var_f, lower=True)
    # the covariance is already extracted
    post_chol = np.sqrt(post_cov)
    # post_chol = cholesky(post_cov)
    # print(post_chol.shape)
    # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to latent dist.
    
    # num_latent x num_latent
    # print(post_chol.shape, post_mean.shape)
    sigma_points = post_chol * random.normal(key=train_rng, shape=(post_chol.shape[0], num_samples)) + post_mean
    # pre-compute wᵢ log p(yₙ|xᵢ√(2vₙ) + mₙ)
    # weighted_log_likelihood_eval = w[:, None] * likelihood.evaluate_log_likelihood(y, sigma_points)
    # Compute expected log likelihood via Monte Carlo:
    # E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) N(fₙ|mₙ,vₙ) dfₙ
    #                 ≈ ∑ᵢ wᵢ log p(yₙ|fsigᵢ)
    exp_log_lik = likelihood.evaluate_log_likelihood(y, sigma_points)
    if missing_mask is not None:
        missing_mask = np.squeeze(missing_mask)
        # return_samples is primarily used to compute test NLL
        # so we don't do any masking in that case
        if return_samples is False:
            exp_log_lik = np.where(missing_mask, exp_log_lik, 0)
    if len(exp_log_lik.shape) > 0:
        if return_samples is False:
            exp_log_lik = np.mean(exp_log_lik, 0)
            exp_log_lik = np.sum(exp_log_lik)
        else:
            exp_log_lik = np.sum(exp_log_lik, [i+1 for i in range(len(exp_log_lik.shape[1:]))])

    return exp_log_lik

def test_loglikelihood_monte_carlo(likelihood, y, post_mean, post_cov, train_rng, missing_mask, num_samples=1, return_samples=False, spatiotemporal=False):
    # num_latent x num_latent
    # post_cov = (post_cov + post_cov.T) / 2
    # chol_f, low = cho_factor(var_f, lower=True)
    # the covariance is already extracted
    post_chol = np.sqrt(post_cov)
    # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to latent dist.
    
    # num_latent x num_latent
    sigma_points = post_chol * random.normal(key=train_rng, shape=(post_chol.shape[0], num_samples)) + post_mean
    exp_log_lik = likelihood.evaluate_log_likelihood(y, sigma_points)
    if return_samples is False:
        exp_log_lik = np.mean(exp_log_lik, axis=0)
    if missing_mask is not None:
        missing_mask = np.squeeze(missing_mask)
        exp_log_lik = np.where(missing_mask, exp_log_lik, 0)
    if len(exp_log_lik.shape) > 0:
        if return_samples:
            if spatiotemporal is False:
                exp_log_lik = np.sum(exp_log_lik, [i+1 for i in range(len(exp_log_lik.shape[1:]))])
        else:
            exp_log_lik = np.sum(exp_log_lik)
    return exp_log_lik
    
    # loglik = likelihood.evaluate_log_likelihood(y, sigma_points)
    # loglik = np.mean(loglik, axis=0)
    # if missing_mask is not None:
    #     missing_mask = np.squeeze(missing_mask)
    #     loglik = np.where(missing_mask, loglik, 0)
    # if len(loglik.shape) > 0:
    #     loglik = np.mean(loglik)
    # return loglik