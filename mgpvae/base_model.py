import objax
import jax.numpy as np
from jax import vmap, random

class BaseGPVAEExternalModel(objax.Module):
    """
    The parent model class: initialises all the common model features and implements shared methods
    """

    def __init__(self, kernel, likelihood, func_dim=1):
        super().__init__()
        self.kernel = kernel
        self.likelihood = likelihood
        self.func_dim = func_dim  # number of latent dimensions
        # as the latent dimensions

    def __call__(self, X=None):
        if X is None:
            self.update_posterior()
        else:
            return self.predict(X)

    def prior_sample(self, num_samps=1, X=None, seed=0):
        raise NotImplementedError

    def update_posterior(self):
        raise NotImplementedError

    def compute_log_lik(self, pseudo_y=None, pseudo_var=None):
        """Compute the log likelihood of the pseudo model, i.e. the log normaliser of the approximate posterior"""
        raise NotImplementedError

    def predict_y(self, X_train, X_test, Y_train, input_mask=None, seed=0, num_samples=1):
        """
        predict y at new test locations X
        """
        mean_f, var_f = self.predict(X_train, X_test, Y_train, input_mask)
        mean_f = mean_f.reshape(mean_f.shape[0], -1, 1)
        var_f = var_f.reshape(var_f.shape[0], -1, 1)
        pred_rng = random.PRNGKey(seed)
        pred_rng = random.split(pred_rng, mean_f.shape[0])
        mean_y, var_y = vmap(self.likelihood.predict, (0, 0, 0, None))(
            mean_f, var_f, pred_rng, num_samples
        )
        return np.squeeze(mean_y), np.squeeze(var_y), mean_f, var_f

    def conditional_data_to_posterior(self, mean_f, cov_f):
        return mean_f, cov_f

    def compute_full_pseudo_lik(self):
        return self.pseudo_likelihood()
