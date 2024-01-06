from ctypes.wintypes import HACCEL
import objax
import jax.numpy as np
from jax import vmap
from jax.scipy.linalg import expm, block_diag
from mgpvae.util import (
    scaled_squared_euclid_dist,
    softplus_inv,
    rotation_matrix,
    tile_arrays,
    cho_factor,
    cho_solve
)
from objax.functional import softplus
from jax.scipy.linalg import cholesky


class Kernel(objax.Module):
    """ """

    def __call__(self, X, X2):
        return self.K(X, X2)

    def K(self, X, X2):
        raise NotImplementedError("kernel function not implemented")

    def measurement_model(self):
        raise NotImplementedError

    def inducing_precision(self):
        return None, None

    def kernel_to_state_space(self, R=None):
        raise NotImplementedError

    def spatial_conditional(self, R=None, predict=False):
        """ """
        return None, None

    def get_meanfield_block_index(self):
        raise Exception(
            "Either the mean-field method is not applicable to this kernel, "
            "or this kernel's get_meanfield_block_index() method has not been implemented"
        )

    def feedback_matrix(self):
        raise NotImplementedError

    def state_transition(self, dt):
        F = self.feedback_matrix()
        A = expm(F * dt)
        return A


class StationaryKernel(Kernel):
    """ """

    def __init__(
        self, variance=1.0, lengthscale=1.0, fix_variance=False, fix_lengthscale=False
    ):
        # check whether the parameters are to be optimised
        if fix_lengthscale:
            self.transformed_lengthscale = objax.StateVar(
                softplus_inv(np.array(lengthscale))
            )
        else:
            self.transformed_lengthscale = objax.TrainVar(
                softplus_inv(np.array(lengthscale))
            )
        if fix_variance:
            self.transformed_variance = objax.StateVar(softplus_inv(np.array(variance)))
        else:
            self.transformed_variance = objax.TrainVar(softplus_inv(np.array(variance)))
        if len(self.transformed_lengthscale.shape) == 0:
            self.num_dim = 0
        else:
            assert (
                variance.shape == lengthscale.shape
            ), "variance and lengthscale shapes need to match"
            self.num_dim = self.lengthscale.shape[0]

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def lengthscale(self):
        return softplus(self.transformed_lengthscale.value)

    def K(self, X, X2):
        r2 = scaled_squared_euclid_dist(X, X2, self.lengthscale)
        return self.K_r2(r2)

    def K_r2(self, r2):
        # Clipping around the (single) float precision which is ~1e-45.
        r = np.sqrt(np.maximum(r2, 1e-36))
        return self.K_r(r)

    @staticmethod
    def K_r(r):
        raise NotImplementedError("kernel not implemented")

    def kernel_to_state_space(self, R=None):
        raise NotImplementedError

    def measurement_model(self):
        raise NotImplementedError

    def stationary_covariance(self):
        raise NotImplementedError

    def feedback_matrix(self):
        raise NotImplementedError


class Matern32(StationaryKernel):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    @property
    def state_dim(self):
        return 2

    def K_r(self, r):
        sqrt3 = np.sqrt(3.0)
        return self.variance * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)

    def kernel_to_state_space_(self, lengthscale, variance):
        lam = 3.0**0.5 / lengthscale
        F = np.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        L = np.array([[0], [1]])
        Qc = np.array([[12.0 * 3.0**0.5 / lengthscale**3.0 * variance]])
        H = np.array([[1.0, 0.0]])
        Pinf = np.array([[variance, 0.0], [0.0, 3.0 * variance / lengthscale**2.0]])
        return F, L, Qc, H, Pinf

    def kernel_to_state_space(self):
        lengthscale = self.lengthscale
        variance = self.variance
        return vmap(self.kernel_to_state_space_, (0, 0))(lengthscale, variance)

    def stationary_covariance(self):
        if self.num_dim == 0:
            return self.stationary_covariance_(
                self.variance,
                self.lengthscale,
            )
        else:
            return vmap(self.stationary_covariance_, (0, 0))(
                self.variance,
                self.lengthscale,
            )

    def stationary_covariance_(self, variance, lengthscale):
        Pinf = np.array([[variance, 0.0], [0.0, 3.0 * variance / lengthscale**2.0]])
        return Pinf

    def measurement_model(self):
        H = np.array([[1.0, 0.0]])
        if self.num_dim == 0:
            return H
        else:
            return np.tile(H[None], [self.variance.shape[0], 1, 1])

    def state_transition(self, dt):
        if self.num_dim == 0:
            return self.state_transition_(
                dt,
                self.lengthscale,
            )
        else:
            return vmap(self.state_transition_, (None, 0))(
                dt,
                self.lengthscale,
            )

    def state_transition_(self, dt, lengthscale):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [2, 2]
        """
        lam = np.sqrt(3.0) / lengthscale
        A = np.exp(-dt * lam) * (
            dt * np.array([[lam, 1.0], [-(lam**2.0), -lam]]) + np.eye(2)
        )
        return A

    def feedback_matrix(self):
        lam = 3.0**0.5 / self.lengthscale
        F = np.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        return F


class Matern52(StationaryKernel):
    """
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
    differentiable. The kernel equation is
    k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}
    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    @property
    def state_dim(self):
        return 3

    def K_r(self, r):
        sqrt5 = np.sqrt(5.0)
        return (
            self.variance
            * (1.0 + sqrt5 * r + 5.0 / 3.0 * np.square(r))
            * np.exp(-sqrt5 * r)
        )

    def stationary_covariance(self):
        if self.num_dim == 0:
            return self.stationary_covariance_(
                self.variance,
                self.lengthscale,
            )
        else:
            return vmap(self.stationary_covariance_, (0, 0))(
                self.variance,
                self.lengthscale,
            )

    def stationary_covariance_(self, variance, lengthscale):
        kappa = 5.0 / 3.0 * variance / lengthscale**2.0
        Pinf = np.array(
            [
                [variance, 0.0, -kappa],
                [0.0, kappa, 0.0],
                [-kappa, 0.0, 25.0 * variance / lengthscale**4.0],
            ]
        )
        return Pinf

    def measurement_model(self):
        H = np.array([[1.0, 0.0, 0.0]])
        if self.num_dim == 0:
            return H
        else:
            return np.tile(H[None], [self.variance.shape[0], 1, 1])

    def state_transition(self, dt):
        if self.num_dim == 0:
            return self.state_transition_(
                dt,
                self.lengthscale,
            )
        else:
            return vmap(self.state_transition_, (None, 0))(
                dt,
                self.lengthscale,
            )

    def state_transition_(self, dt, lengthscale):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-5/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [3, 3]
        """
        lam = np.sqrt(5.0) / lengthscale
        dtlam = dt * lam
        A = np.exp(-dtlam) * (
            dt
            * np.array(
                [
                    [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                    [-0.5 * dtlam * lam**2, lam * (1.0 - dtlam), 1.0 - 0.5 * dtlam],
                    [
                        lam**3 * (0.5 * dtlam - 1.0),
                        lam**2 * (dtlam - 3),
                        lam * (0.5 * dtlam - 2.0),
                    ],
                ]
            )
            + np.eye(3)
        )
        return A

class QuasiPeriodicMatern32(Kernel):
    """
    Quasi-periodic kernel in SDE form (product of Periodic and Matern-3/2).
    Hyperparameters:
        variance, σ²
        lengthscale of Periodic, l_p
        period, p
        lengthscale of Matern, l_m
    The associated continuous-time state space model matrices are constructed via
    a sum of cosines times a Matern-3/2.
    """

    def __init__(
        self,
        variance=1.0,
        lengthscale_periodic=1.0,
        period=1.0,
        lengthscale_matern=1.0,
        order=6,
    ):
        self.transformed_lengthscale_periodic = objax.TrainVar(
            np.array(softplus_inv(lengthscale_periodic))
        )
        self.transformed_variance = objax.TrainVar(np.array(softplus_inv(variance)))
        self.transformed_period = objax.TrainVar(np.array(softplus_inv(period)))
        self.transformed_lengthscale_matern = objax.TrainVar(
            np.array(softplus_inv(lengthscale_matern))
        )
        super().__init__()
        self.name = "Quasi-periodic Matern-3/2"
        self.order = order
        self.igrid = np.meshgrid(np.arange(self.order + 1), np.arange(self.order + 1))[
            1
        ]
        factorial_mesh_K = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0],
                [120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
                [720.0, 720.0, 720.0, 720.0, 720.0, 720.0, 720.0],
            ]
        )
        b = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [6.0, 0.0, 8.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 20.0, 0.0, 10.0, 0.0, 2.0, 0.0],
                [20.0, 0.0, 30.0, 0.0, 12.0, 0.0, 2.0],
            ]
        )
        self.b_fmK_2igrid = b * (1.0 / factorial_mesh_K) * (2.0**-self.igrid)

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def lengthscale_periodic(self):
        return softplus(self.transformed_lengthscale_periodic.value)

    @property
    def lengthscale_matern(self):
        return softplus(self.transformed_lengthscale_matern.value)

    @property
    def period(self):
        return softplus(self.transformed_period.value)

    def K(self, X, X2):
        raise NotImplementedError

    def kernel_to_state_space(self):
        return vmap(self.kernel_to_state_space_, (0, 0, 0, 0))(
            self.lengthscale_periodic,
            self.lengthscale_matern,
            self.variance,
            self.period,
        )

    def kernel_to_state_space_(
        self, lengthscale_periodic, lengthscale_matern, variance, period
    ):
        var_p = 1.0
        ell_p = lengthscale_periodic
        a = (
            self.b_fmK_2igrid
            * ell_p ** (-2.0 * self.igrid)
            * np.exp(-1.0 / ell_p**2.0)
            * var_p
        )
        q2 = np.sum(a, axis=0)
        # The angular frequency
        omega = 2 * np.pi / period
        # The model
        F_p = np.kron(
            np.diag(np.arange(self.order + 1)), np.array([[0.0, -omega], [omega, 0.0]])
        )
        L_p = np.eye(2 * (self.order + 1))
        # Qc_p = np.zeros(2 * (self.N + 1))
        Pinf_p = np.kron(np.diag(q2), np.eye(2))
        H_p = np.kron(np.ones([1, self.order + 1]), np.array([1.0, 0.0]))
        lam = 3.0**0.5 / lengthscale_matern
        F_m = np.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        L_m = np.array([[0], [1]])
        Qc_m = np.array([[12.0 * 3.0**0.5 / lengthscale_matern**3.0 * variance]])
        H_m = np.array([[1.0, 0.0]])
        Pinf_m = np.array(
            [
                [self.variance, 0.0],
                [0.0, 3.0 * self.variance / lengthscale_matern**2.0],
            ]
        )
        # F = np.kron(F_p, np.eye(2)) + np.kron(np.eye(14), F_m)
        F = np.kron(F_m, np.eye(2 * (self.order + 1))) + np.kron(np.eye(2), F_p)
        L = np.kron(L_m, L_p)
        Qc = np.kron(Qc_m, Pinf_p)
        H = np.kron(H_m, H_p)
        # Pinf = np.kron(Pinf_m, Pinf_p)
        Pinf = block_diag(
            np.kron(Pinf_m, q2[0] * np.eye(2)),
            np.kron(Pinf_m, q2[1] * np.eye(2)),
            np.kron(Pinf_m, q2[2] * np.eye(2)),
            np.kron(Pinf_m, q2[3] * np.eye(2)),
            np.kron(Pinf_m, q2[4] * np.eye(2)),
            np.kron(Pinf_m, q2[5] * np.eye(2)),
            np.kron(Pinf_m, q2[6] * np.eye(2)),
        )
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        return vmap(self.stationary_covariance_, (0, 0, 0))(
            self.lengthscale_periodic, self.lengthscale_matern, self.variance
        )

    def stationary_covariance_(
        self, lengthscale_periodic, lengthscale_matern, variance
    ):
        var_p = 1.0
        ell_p = lengthscale_periodic
        a = (
            self.b_fmK_2igrid
            * ell_p ** (-2.0 * self.igrid)
            * np.exp(-1.0 / ell_p**2.0)
            * var_p
        )
        q2 = np.sum(a, axis=0)
        Pinf_m = np.array(
            [[variance, 0.0], [0.0, 3.0 * variance / lengthscale_matern**2.0]]
        )
        Pinf = block_diag(
            np.kron(Pinf_m, q2[0] * np.eye(2)),
            np.kron(Pinf_m, q2[1] * np.eye(2)),
            np.kron(Pinf_m, q2[2] * np.eye(2)),
            np.kron(Pinf_m, q2[3] * np.eye(2)),
            np.kron(Pinf_m, q2[4] * np.eye(2)),
            np.kron(Pinf_m, q2[5] * np.eye(2)),
            np.kron(Pinf_m, q2[6] * np.eye(2)),
        )
        return Pinf

    def measurement_model(self):
        H_p = np.kron(np.ones([1, self.order + 1]), np.array([1.0, 0.0]))
        H_m = np.array([[1.0, 0.0]])
        H = np.kron(H_m, H_p)
        H = np.tile(H[None], [self.lengthscale_periodic.shape[0], 1, 1])
        return H

    def state_transition(self, dt):
        return vmap(self.state_transition_, (None, 0, 0, None))(
            dt, self.lengthscale_matern, self.period, self.order
        )

    def state_transition_(self, dt, lengthscale_matern, period, order):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Quasi-Periodic Matern-3/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [M+1, 1]
        :return: state transition matrix A [M+1, D, D]
        """
        lam = np.sqrt(3.0) / lengthscale_matern
        # The angular frequency
        omega = 2 * np.pi / period
        harmonics = np.arange(order + 1) * omega
        R0 = self.subband_mat32(dt, lam, harmonics[0])
        R1 = self.subband_mat32(dt, lam, harmonics[1])
        R2 = self.subband_mat32(dt, lam, harmonics[2])
        R3 = self.subband_mat32(dt, lam, harmonics[3])
        R4 = self.subband_mat32(dt, lam, harmonics[4])
        R5 = self.subband_mat32(dt, lam, harmonics[5])
        R6 = self.subband_mat32(dt, lam, harmonics[6])
        A = np.exp(-dt * lam) * block_diag(R0, R1, R2, R3, R4, R5, R6)
        return A

    @staticmethod
    def subband_mat32(dt, lam, omega):
        R = rotation_matrix(dt, omega)
        Ri = np.block(
            [[(1.0 + dt * lam) * R, dt * R], [-dt * lam**2 * R, (1.0 - dt * lam) * R]]
        )
        return Ri

    def feedback_matrix(self):
        # The angular frequency
        omega = 2 * np.pi / self.period
        # The model
        F_p = np.kron(
            np.diag(np.arange(self.order + 1)), np.array([[0.0, -omega], [omega, 0.0]])
        )
        lam = 3.0**0.5 / self.lengthscale_matern
        F_m = np.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        F = np.kron(F_m, np.eye(2 * (self.order + 1))) + np.kron(np.eye(2), F_p)
        return F


class Periodic(Kernel):
    """
    Periodic kernel in SDE form.
    Hyperparameters:
        variance, σ²
        lengthscale, l
        period, p
    The associated continuous-time state space model matrices are constructed via
    a sum of cosines.
    TODO: allow for orders other than 6

    Doesn't work for some reason.
    """

    def __init__(
        self, variance=1.0, lengthscale=1.0, period=1.0, order=6, fix_variance=False
    ):
        self.transformed_lengthscale = objax.TrainVar(
            np.array(softplus_inv(lengthscale))
        )
        if fix_variance:
            self.transformed_variance = objax.StateVar(np.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(np.array(softplus_inv(variance)))
        self.transformed_period = objax.TrainVar(np.array(softplus_inv(period)))
        super().__init__()
        self.name = "Periodic"
        self.order = order
        self.M = np.meshgrid(np.arange(self.order + 1), np.arange(self.order + 1))[1]
        factorial_mesh_M = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0],
                [120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
                [720.0, 720.0, 720.0, 720.0, 720.0, 720.0, 720.0],
            ]
        )
        b = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [6.0, 0.0, 8.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 20.0, 0.0, 10.0, 0.0, 2.0, 0.0],
                [20.0, 0.0, 30.0, 0.0, 12.0, 0.0, 2.0],
            ]
        )
        self.b_fmK_2M = b * (1.0 / factorial_mesh_M) * (2.0**-self.M)
        if len(self.transformed_lengthscale.shape) == 0:
            self.num_dim = 0
        else:
            self.num_dim = self.lengthscale.shape[0]

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def lengthscale(self):
        return softplus(self.transformed_lengthscale.value)

    @property
    def period(self):
        return softplus(self.transformed_period.value)

    def kernel_to_state_space(self):
        a = (
            self.b_fmK_2M
            * self.lengthscale ** (-2.0 * self.M)
            * np.exp(-1.0 / self.lengthscale**2.0)
            * self.variance
        )
        q2 = np.sum(a, axis=0)
        # The angular frequency
        omega = 2 * np.pi / self.period
        # The model
        F = np.kron(
            np.diag(np.arange(self.order + 1)), np.array([[0.0, -omega], [omega, 0.0]])
        )
        L = np.eye(2 * (self.order + 1))
        Qc = np.zeros(2 * (self.order + 1))
        Pinf = np.kron(np.diag(q2), np.eye(2))
        H = np.kron(np.ones([1, self.order + 1]), np.array([1.0, 0.0]))
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        if self.num_dim == 0:
            return self.stationary_covariance_(self.lengthscale, self.variance)
        else:
            return vmap(self.stationary_covariance_, (0, 0))(
                self.lengthscale,
                self.variance,
            )

    def stationary_covariance_(self, lengthscale, variance):
        a = (
            self.b_fmK_2M
            * lengthscale ** (-2.0 * self.M)
            * np.exp(-1.0 / lengthscale**2.0)
            * variance
        )
        q2 = np.sum(a, axis=0)
        Pinf = np.kron(np.diag(q2), np.eye(2))
        return Pinf

    def measurement_model(self):
        H = np.kron(np.ones([1, self.order + 1]), np.array([1.0, 0.0]))
        if self.num_dim == 0:
            return H
        else:
            H = np.tile(H[None], [self.variance.shape[0], 1, 1])
            return H

    def state_transition(self, dt):
        if self.num_dim == 0:
            return self.state_transition_(self.lengthscale, self.period)
        else:
            return vmap(self.state_transition_, (None, 0))(dt, self.period)

    def state_transition_(self, dt, period):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Periodic prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :return: state transition matrix A [2(N+1), 2(N+1)]
        """
        omega = 2 * np.pi / period  # The angular frequency
        harmonics = np.arange(self.order + 1) * omega
        R0 = rotation_matrix(dt, harmonics[0])
        R1 = rotation_matrix(dt, harmonics[1])
        R2 = rotation_matrix(dt, harmonics[2])
        R3 = rotation_matrix(dt, harmonics[3])
        R4 = rotation_matrix(dt, harmonics[4])
        R5 = rotation_matrix(dt, harmonics[5])
        R6 = rotation_matrix(dt, harmonics[6])
        A = np.block(
            [
                [R0, np.zeros([2, 12])],
                [np.zeros([2, 2]), R1, np.zeros([2, 10])],
                [np.zeros([2, 4]), R2, np.zeros([2, 8])],
                [np.zeros([2, 6]), R3, np.zeros([2, 6])],
                [np.zeros([2, 8]), R4, np.zeros([2, 4])],
                [np.zeros([2, 10]), R5, np.zeros([2, 2])],
                [np.zeros([2, 12]), R6],
            ]
        )
        return A

    def feedback_matrix(self):
        # The angular frequency
        omega = 2 * np.pi / self.period
        # The model
        F = np.kron(
            np.diag(np.arange(self.order + 1)), np.array([[0.0, -omega], [omega, 0.0]])
        )
        return F


class SubbandMatern32(Kernel):
    """
    Subband Matern-3/2 kernel in SDE form (product of Cosine and Matern-3/2).
    Hyperparameters:
        variance, σ²
        lengthscale, l
        radial frequency, ω
    The associated continuous-time state space model matrices are constructed via
    kronecker sums and products of the Matern3/2 and cosine components:
    letting λ = √3 / l
    F      = F_mat3/2 ⊕ F_cos  =  ( 0     -ω     1     0
                                    ω      0     0     1
                                   -λ²     0    -2λ   -ω
                                    0     -λ²    ω    -2λ )
    L      = L_mat3/2 ⊗ I      =  ( 0      0
                                    0      0
                                    1      0
                                    0      1 )
    Qc     = I ⊗ Qc_mat3/2     =  ( 4λ³σ²  0
                                    0      4λ³σ² )
    H      = H_mat3/2 ⊗ H_cos  =  ( 1      0     0      0 )
    Pinf   = Pinf_mat3/2 ⊗ I   =  ( σ²     0     0      0
                                    0      σ²    0      0
                                    0      0     3σ²/l² 0
                                    0      0     0      3σ²/l²)
    and the discrete-time transition matrix is (for step size Δt),
    R = ( cos(ωΔt)   -sin(ωΔt)
          sin(ωΔt)    cos(ωΔt) )
    A = exp(-Δt/l) ( (1+Δtλ)R   ΔtR
                     -Δtλ²R    (1-Δtλ)R )
    """

    def __init__(
        self, variance=1.0, lengthscale=1.0, radial_frequency=1.0, fix_variance=False
    ):
        self.transformed_lengthscale = objax.TrainVar(
            np.array(softplus_inv(lengthscale))
        )
        if fix_variance:
            self.transformed_variance = objax.StateVar(np.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(np.array(softplus_inv(variance)))
        self.transformed_radial_frequency = objax.TrainVar(
            np.array(softplus_inv(radial_frequency))
        )
        super().__init__()
        self.name = "Subband Matern-3/2"

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def lengthscale(self):
        return softplus(self.transformed_lengthscale.value)

    @property
    def radial_frequency(self):
        return softplus(self.transformed_radial_frequency.value)

    def K(self, X, X2):
        raise NotImplementedError

    def kernel_to_state_space(self):
        return vmap(self.kernel_to_state_space_, (0, 0, 0, 0))(
            self.lengthscale, self.variance, self.radial_frequency
        )

    def kernel_to_state_space_(self, lengthscale, variance, radial_frequency):
        lam = 3.0**0.5 / lengthscale
        F_mat = np.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        L_mat = np.array([[0], [1]])
        Qc_mat = np.array([[12.0 * 3.0**0.5 / lengthscale**3.0 * variance]])
        H_mat = np.array([[1.0, 0.0]])
        Pinf_mat = np.array(
            [[variance, 0.0], [0.0, 3.0 * variance / lengthscale**2.0]]
        )
        F_cos = np.array([[0.0, -radial_frequency], [radial_frequency, 0.0]])
        H_cos = np.array([[1.0, 0.0]])
        # F = (0   -ω   1   0
        #      ω    0   0   1
        #      -λ²  0  -2λ -ω
        #      0   -λ²  ω  -2λ)
        F = np.kron(F_mat, np.eye(2)) + np.kron(np.eye(2), F_cos)
        L = np.kron(L_mat, np.eye(2))
        Qc = np.kron(np.eye(2), Qc_mat)
        H = np.kron(H_mat, H_cos)
        Pinf = np.kron(Pinf_mat, np.eye(2))
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        return vmap(self.stationary_covariance_, (0, 0))(
            self.variance,
            self.lengthscale,
        )

    def stationary_covariance_(self, variance, lengthscale):
        Pinf_mat = np.array(
            [[variance, 0.0], [0.0, 3.0 * variance / lengthscale**2.0]]
        )
        Pinf = np.kron(Pinf_mat, np.eye(2))
        return Pinf

    def measurement_model(self):
        H_mat = np.array([[1.0, 0.0]])
        H_cos = np.array([[1.0, 0.0]])
        H = np.kron(H_mat, H_cos)
        H = np.tile(H[None], [self.variance.shape[0], 1, 1])
        return H

    def state_transition(self, dt):
        return vmap(self.state_transition_, (None, 0, 0))(
            dt, self.lengthscale, self.radial_frequency
        )

    def state_transition_(self, dt, lengthscale, radial_frequency):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Subband Matern-3/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :return: state transition matrix A [4, 4]
        """
        lam = np.sqrt(3.0) / lengthscale
        R = rotation_matrix(dt, radial_frequency)
        A = np.exp(-dt * lam) * np.block(
            [[(1.0 + dt * lam) * R, dt * R], [-dt * lam**2 * R, (1.0 - dt * lam) * R]]
        )
        return A

    def feedback_matrix(self):
        lam = 3.0**0.5 / self.lengthscale
        F_mat = np.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        F_cos = np.array([[0.0, -self.radial_frequency], [self.radial_frequency, 0.0]])
        # F = (0   -ω   1   0
        #      ω    0   0   1
        #      -λ²  0  -2λ -ω
        #      0   -λ²  ω  -2λ)
        F = np.kron(F_mat, np.eye(2)) + np.kron(np.eye(2), F_cos)
        return F


class MeanfieldKernel(Kernel):
    """Meanfield kernel i.e. same kernel for all latent dimensions"""

    def __init__(self, base_kernel: Kernel, num_latent: int) -> None:
        self.base_kernel = base_kernel
        self.num_latent = num_latent

    def kernel_to_state_space(self):
        res = self.base_kernel.kernel_to_state_space()
        return tile_arrays(res, self.num_latent)

    def stationary_covariance(self):
        res = self.base_kernel.stationary_covariance()
        return tile_arrays(res, self.num_latent)[0]

    def measurement_model(self):
        res = self.base_kernel.measurement_model()
        return tile_arrays(res, self.num_latent)[0]

    def state_transition(self, dt):
        res = self.base_kernel.state_transition(dt)
        return tile_arrays(res, self.num_latent)[0]


class Cosine(Kernel):
    """
    Cosine kernel in SDE form.
    Hyperparameters:
        radial frequency, ω
    The associated continuous-time state space model matrices are:
    F      = ( 0   -ω
               ω    0 )
    L      = N/A
    Qc     = N/A
    H      = ( 1  0 )
    Pinf   = ( 1  0
               0  1 )
    and the discrete-time transition matrix is (for step size Δt),
    A      = ( cos(ωΔt)   -sin(ωΔt)
               sin(ωΔt)    cos(ωΔt) )
    """

    def __init__(self, frequency=1.0):
        self.transformed_frequency = objax.TrainVar(np.array(softplus_inv(frequency)))
        super().__init__()
        self.name = "Cosine"

    @property
    def frequency(self):
        return softplus(self.transformed_frequency.value)

    def kernel_to_state_space(self, R=None):
        F = np.array([[0.0, -self.frequency], [self.frequency, 0.0]])
        H = np.array([[1.0, 0.0]])
        L = []
        Qc = []
        Pinf = np.eye(2)
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        Pinf = np.eye(2)
        Pinf = np.tile(Pinf[None], [self.transformed_frequency.shape[0], 1, 1])
        return Pinf

    def measurement_model(self):
        H = np.array([[1.0, 0.0]])
        H = np.tile(H[None], [self.transformed_frequency.shape[0], 1, 1])
        return H

    def state_transition(self, dt):
        return vmap(self.state_transition_, (None, 0))(dt, self.frequency)

    def state_transition_(self, dt, frequency):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Cosine prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [M+1, 1]
        :return: state transition matrix A [M+1, D, D]
        """
        state_transitions = rotation_matrix(dt, frequency)  # [2, 2]
        return state_transitions

    def feedback_matrix(self):
        F = np.array([[0.0, -self.frequency], [self.frequency, 0.0]])
        return


## spatiotemporal


class SpatiotemporalMatern32(StationaryKernel):
    """
    The Spatiotemporal Matern 3/2 - Matern3/2 kernel (space and time). 
    """

    def __init__(
        self,
        R,
        lengthscale=1,
        lengthscale_time=1,
        variance=1,
        variance_time=1,
        fix_variance=False,
        fix_lengthscale=False,
    ):
        super().__init__(variance, lengthscale, fix_variance, fix_lengthscale)
        # fixed spatial locations
        self.R = R  # shape Ns x 2
        self.Ns = R.shape[0]
        self.transformed_lengthscale_time = objax.TrainVar(softplus_inv(np.array(lengthscale_time)))
        self.transformed_variance_time = objax.TrainVar(softplus_inv(np.array(variance_time)))
        self.num_dim = self.lengthscale_time.shape[0]
        state = np.ones([self.state_dim, self.state_dim])
        for _ in range(1, self.Ns):
            state = block_diag(state, np.ones([self.state_dim, self.state_dim]))
        block_index = np.where(np.array(state, dtype=bool))
        self.block_index = block_index
    
    @property
    def variance_time(self):
        return softplus(self.transformed_lengthscale_time.value)

    @property
    def lengthscale_time(self):
        return softplus(self.transformed_variance_time.value)
    
    @property
    def state_dim(self):
        return 2

    def K_r(self, r):
        sqrt3 = np.sqrt(3.0)
        return (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)
    def K_space(self, X, X2):
        r2 = scaled_squared_euclid_dist(X, X2, self.lengthscale)
        return self.K_r2(r2)
    def K_time_(self, X, X2, lengthscale_time, variance_time):
        r2 = scaled_squared_euclid_dist(X, X2, lengthscale_time)
        return self.K_r2(r2) * variance_time
    def K_time(self, X, X2):
        return vmap(self.K_time_, (None, None, 0, 0))(X, X2, self.lengthscale_time, self.variance_time)
    def spatial_conditional(self, X=None, R=None):
        """
        Compute the spatial conditional, i.e. the measurement model projecting the latent function u(t) to f(X,R)
            f(X,R) | u(t) ~ N(f(X,R) | B u(t), C)
        """
        Qss, Lss = self.inducing_precision() # pre-calculate inducing precision and its Cholesky factor
        Krz = self.K_space(R, self.R)
        K = Krz @ Qss  # Krz / Kzz
        B = K @ Lss
        C = vmap(self.full_conditional, (0, None, None, None))(X, R, Krz, K)  # conditional covariance. Repeat in num_latent
        return B, C
    def full_conditional(self, X, R, Krz, K):
        Krr = self.K_space(R, R)
        X = X.reshape(-1, 1)
        cov = self.K_time(X, X) * (Krr - K @ Krz.T)
        return cov
    def inducing_precision(self):
        """
        Compute the covariance and precision of the inducing spatial points to be used during filtering
        """
        Kss = self.K_space(self.R, self.R) + np.eye(self.Ns)*1e-8 # add conditioning
        Lss, low = cho_factor(Kss, lower=True)  # K_RR^(1/2)
        Qss = cho_solve((Lss, low), np.eye(self.Ns))  # K_RR^(-1)
        return Qss, Lss
    def precompute_spatial_mixing(self):
        """
        Compute the cholesky decomposition of the spatial mixing matrix
        """
        # TODO: add unit testing to check for equivalence with Kss
        Kss = self.K(self.R, self.R) + np.eye(self.Ns)*1e-8 # add conditioning
        # should we be using lower=True?
        Lss = cholesky(Kss, lower=True)
        self.Lss = Lss

    def stationary_covariance(self):
        if self.num_dim == 0:
            return self.stationary_covariance_(
                self.variance_time,
                self.lengthscale_time,
            )
        else:
            return vmap(self.stationary_covariance_, (0, 0))(
                self.variance_time,
                self.lengthscale_time,
            )


    def stationary_covariance_(self, variance_time, lengthscale_time):
        Pinf = np.array([[variance_time, 0.0], [0.0, 3.0 * variance_time / lengthscale_time**2.0]])
        Pinf = np.tile(Pinf, [self.Ns, 1, 1])
        return Pinf

    def stationary_covariance_stacked(self):
        if self.num_dim == 0:
            return self.stationary_covariance_stacked_(
                self.variance_time,
                self.lengthscale_time,
            )
        else:
            return vmap(self.stationary_covariance_stacked_, (0, 0))(
                self.variance_time,
                self.lengthscale_time,
            )
    def stationary_covariance_stacked_(self, variance_time, lengthscale_time):
        Pinf = np.array([[variance_time, 0.0], [0.0, 3.0 * variance_time / lengthscale_time**2.0]])
        Pinf = np.kron(np.eye(self.Ns), Pinf)
        return Pinf

    def measurement_model(self):
        H = np.array([[1.0, 0.0]])
        H = np.kron(np.eye(self.Ns), H)
        # H = np.kron(self.Lss, H)
        if self.num_dim == 0:
            return H
        else:
            return np.tile(H[None], [self.variance_time.shape[0], 1, 1])

    def state_transition(self, dt):
        if self.num_dim == 0:
            return self.state_transition_(
                dt,
                self.lengthscale_time,
            )
        else:
            return vmap(self.state_transition_, (None, 0))(
                dt,
                self.lengthscale_time,
            )

    def state_transition_(self, dt, lengthscale_time):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [2, 2]
        """
        lam = np.sqrt(3.0) / lengthscale_time
        A = np.exp(-dt * lam) * (
            dt * np.array([[lam, 1.0], [-(lam**2.0), -lam]]) + np.eye(2)
        )
        A = np.tile(A, [self.Ns, 1, 1])
        return A

    def state_transition_stacked(self, dt):
        if self.num_dim == 0:
            return self.state_transition_stacked_(
                dt,
                self.lengthscale_time,
            )
        else:
            return vmap(self.state_transition_stacked_, (None, 0))(
                dt,
                self.lengthscale_time,
            )

    def state_transition_stacked_(self, dt, lengthscale_time):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [2, 2]
        """
        lam = np.sqrt(3.0) / lengthscale_time
        A = np.exp(-dt * lam) * (
            dt * np.array([[lam, 1.0], [-(lam**2.0), -lam]]) + np.eye(2)
        )
        A = np.kron(np.eye(self.Ns), A)
        return A