import objax 
from objax.nn.init import xavier_normal
from objax.variable import TrainVar
from objax.typing import JaxArray
from objax.util import class_name
from objax import util
from jax import numpy as jn
from typing import Callable, Tuple, Union
from mgpvae.util import softplus
from objax.constants import ConvPadding
from objax.nn.init import kaiming_normal, xavier_normal
from objax.typing import JaxArray, ConvPaddingInt
import jax 
from jax import lax
from jax import random
from mgpvae.util import softplus_inv, softplus
from functools import partial


# Encoder network
class CNNEncoderVAE(objax.Module):
    def __init__(self, nin, nout, num_hidden, num_latent, encoder = None) -> None:
        super().__init__()

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = objax.nn.Sequential([
                Conv2D(nin=nin, nout=32, k=3, strides=(2, 2)), 
                objax.functional.relu,
                Conv2D(nin=32, nout=nout, k=3, strides=(2, 2)), 
                objax.functional.relu,
            ])
        self.hidden_to_mu = objax.nn.Sequential(
                        [
                            Linear(num_hidden * 16 * 16, num_latent),
                        ]
        )
        self.hidden_to_var = objax.nn.Sequential(
                        [
                            Linear(num_hidden * 16 * 16, num_latent),
                        ]
        )
    def __call__(self, x):
        if len(x.shape) == 3:
            out = self.encoder(x[None, :, :, :])
        else:
            out = self.encoder(x)
        out = objax.functional.flatten(out)
        return self.hidden_to_mu(out), self.hidden_to_var(out)
        
# Define the VAE model
class VAE(objax.Module):
    def __init__(self, encoder, decoder, scale=1.0):
        self.encoder = encoder
        self.decoder = decoder
        self.transformed_scale = objax.TrainVar(jn.array(softplus_inv(scale)))

    @property
    def scale(self):
        return softplus(self.transformed_scale.value)

    def forward(self, x, train_rng):
        # Encode the input
        z_mean, z_log_var = self.encoder(x)
        # z_mean, z_log_var = z_mean.astype(jn.float64), z_log_var.astype(jn.float64)
        # Sample from the learned distribution
        eps = random.normal(key=train_rng, shape=z_mean.shape, dtype=z_mean.dtype)
        z = z_mean + jn.exp(0.5 * z_log_var) * eps

        # Decode the sampled representation
        x_hat = self.decoder(z)

        return x_hat, z_mean, z_log_var

    def loss_batch(self, x, train_rng):
        train_rng = random.split(random.PRNGKey(0), x.shape[0])
        elbo_batch = jax.vmap(self.elbo, (0, 0))(x, train_rng)
        return -jn.mean(elbo_batch)
    
    def elbo(self, x, train_rng):
        x_hat, z_mean, z_log_var = self.forward(x, train_rng)
        # Compute the KL-divergence
        kl = - 0.5 * jn.sum(1 + z_log_var - jn.power(z_mean, 2) - jn.exp(z_log_var))
        # Compute the reconstruction loss
        # log_px_z = -jn.sum(0.5*((x - x_hat) / self.scale)**2, axis=[1,2,3]) - jn.log(self.scale)
        log_px_z = -jn.sum(0.5*((x - x_hat) / 0.1)**2, axis=[1,2,3]) - jn.log(0.1)
        # log_px_z = -jn.sum(0.5*(x - x_hat)**2, axis=[1,2,3])
        # Compute the ELBO
        elbo = log_px_z - kl
        return elbo


class DiagonalEncoder(objax.Module):
    def __init__(self) -> None:
        super().__init__()

        

class CNNEncoder(objax.Module):
    """Same architecture as https://www.tensorflow.org/tutorials/generative/cvae
    """

    def __init__(self, nin, nout, encoder = None) -> None:
        super().__init__()

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = objax.nn.Sequential([
                Conv2D(nin=nin, nout=32, k=3, strides=(2, 2)), 
                objax.functional.relu,
                Conv2D(nin=32, nout=nout, k=3, strides=(2, 2)), 
                objax.functional.relu,
            ])

    def __call__(self, x):
        # if the image is a single channel, unsqueeze axis=1
        if len(x.shape) == 3:
            out = self.encoder(x[:, None, :, :])
        else:
            out = self.encoder(x)
        return objax.functional.flatten(out)
        

class CNNDecoder(objax.Module):
    """Same architecture as https://www.tensorflow.org/tutorials/generative/cvae
    """

    def __init__(self, nin, nout, decoder = None, linear1 = None, last_sigmoid=False) -> None:
        super().__init__()

        if linear1:
            self.linear1 = linear1
        else:
            self.linear1 = Linear(nin=nin, nout=8*8*32)

        if decoder:
            self.decoder = decoder
        else:
            self.decoder = objax.nn.Sequential([
                objax.functional.relu,
                ConvTranspose2D(nin=32, nout=64, k=3, strides=2), 
                objax.functional.relu,
                ConvTranspose2D(nin=64, nout=32, k=3, strides=2), 
                objax.functional.relu,
                ConvTranspose2D(nin=32, nout=nout, k=3, strides=1), 
            ])
        self.last_sigmoid = last_sigmoid

    def __call__(self, x):
        out = self.linear1(x)
        # TODO: hardcoded
        out = self.decoder(out.reshape(-1, 32, 8, 8))
        if self.last_sigmoid:
            return objax.functional.sigmoid(out)
        else:
            return out
        


class LinearPositive(objax.Module):
    """Applies a linear transformation on an input batch."""

    def __init__(self, nin: int, nout: int, use_bias: bool = True, w_init: Callable = xavier_normal):
        """Creates a Linear module instance.

        Args:
            nin: number of channels of the input tensor.
            nout: number of channels of the output tensor.
            use_bias: if True then linear layer will have bias term.
            w_init: weight initializer for linear layer (a function that takes in a IO shape and returns a 2D matrix).
        """
        self.w_init = w_init
        self.b = TrainVar(jn.zeros(nout)) if use_bias else None
        self.w = TrainVar(w_init((nin, nout)))

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the linear transformation to input x."""
        y = jn.dot(x, softplus(self.w.value))
        if self.b is not None:
            y += self.b.value
        return y

    def __repr__(self):
        s = self.w.value.shape
        args = f'nin={s[0]}, nout={s[1]}, use_bias={self.b is not None}, w_init={util.repr_function(self.w_init)}'
        return f'{class_name(self)}({args})'

class Conv1D(objax.module.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.w = TrainVar(jn.zeros([1, nin, nout]))
        self.b = TrainVar(jn.zeros([nout]))

    def __call__(self, x):  # NCHW
        *start, nx = x.shape
        assert nx == self.nx

        return jn.reshape(jn.matmul(jn.reshape(x, [-1, self.nin]),
                                    jn.reshape(self.w.value, [-1, self.nout])) + self.b.value,
                          start + [self.nf])

class Conv2D(objax.Module):
    """Applies a 2D convolution on a 4D-input batch of shape (N,C,H,W)."""

    def __init__(self,
                 nin: int,
                 nout: int,
                 k: Union[Tuple[int, int], int],
                 strides: Union[Tuple[int, int], int] = 1,
                 dilations: Union[Tuple[int, int], int] = 1,
                 groups: int = 1,
                 padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.SAME,
                 use_bias: bool = True,
                 w_init: Callable = kaiming_normal):
        """Creates a Conv2D module instance.

        Args:
            nin: number of channels of the input tensor.
            nout: number of channels of the output tensor.
            k: size of the convolution kernel, either tuple (height, width) or single number if they're the same.
            strides: convolution strides, either tuple (stride_y, stride_x) or single number if they're the same.
            dilations: spacing between kernel points (also known as astrous convolution),
                       either tuple (dilation_y, dilation_x) or single number if they're the same.
            groups: number of input and output channels group. When groups > 1 convolution operation is applied
                    individually for each group. nin and nout must both be divisible by groups.
            padding: padding of the input tensor, either Padding.SAME, Padding.VALID or numerical values.
            use_bias: if True then convolution will have bias term.
            w_init: initializer for convolution kernel (a function that takes in a HWIO shape and returns a 4D matrix).
        """
        super().__init__()
        assert nin % groups == 0, 'nin should be divisible by groups'
        assert nout % groups == 0, 'nout should be divisible by groups'
        self.b = TrainVar(jn.zeros((nout, 1, 1), dtype=jn.float32)) if use_bias else None
        self.w = TrainVar(w_init((*util.to_tuple(k, 2), nin // groups, nout)).astype(jn.float32))  # HWIO
        self.padding = util.to_padding(padding, 2)
        self.strides = util.to_tuple(strides, 2)
        self.dilations = util.to_tuple(dilations, 2)
        self.groups = groups
        self.w_init = w_init

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the convolution to input x."""
        nin = self.w.value.shape[2] * self.groups
        assert x.shape[1] == nin, (f'Attempting to convolve an input with {x.shape[1]} input channels '
                                   f'when the convolution expects {nin} channels. For reference, '
                                   f'self.w.value.shape={self.w.value.shape} and x.shape={x.shape}.')
        y = lax.conv_general_dilated(x, self.w.value, self.strides, self.padding,
                                     rhs_dilation=self.dilations,
                                     feature_group_count=self.groups,
                                     dimension_numbers=('NCHW', 'HWIO', 'NCHW'))
        if self.b is not None:
            y += self.b.value
        return y

    def __repr__(self):
        args = dict(nin=self.w.value.shape[2] * self.groups, nout=self.w.value.shape[3], k=self.w.value.shape[:2],
                    strides=self.strides, dilations=self.dilations, groups=self.groups, padding=self.padding,
                    use_bias=self.b is not None)
        args = ', '.join(f'{k}={repr(v)}' for k, v in args.items())
        return f'{class_name(self)}({args}, w_init={util.repr_function(self.w_init)})'

class ConvTranspose2D(Conv2D):
    """Applies a 2D transposed convolution on a 4D-input batch of shape (N,C,H,W).

    This module can be seen as a transformation going in the opposite direction of a normal convolution, i.e.,
    from something that has the shape of the output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with said convolution.
    Note that ConvTranspose2D is consistent with
    `Conv2DTranspose <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose>`_,
    of Tensorflow but is not consistent with
    `ConvTranspose2D <https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose2d.html>`_
    of PyTorch due to kernel transpose and padding.
    """

    def __init__(self,
                 nin: int,
                 nout: int,
                 k: Union[Tuple[int, int], int],
                 strides: Union[Tuple[int, int], int] = 1,
                 dilations: Union[Tuple[int, int], int] = 1,
                 padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.SAME,
                 use_bias: bool = True,
                 w_init: Callable = kaiming_normal):
        """Creates a ConvTranspose2D module instance.

        Args:
            nin: number of channels of the input tensor.
            nout: number of channels of the output tensor.
            k: size of the convolution kernel, either tuple (height, width) or single number if they're the same.
            strides: convolution strides, either tuple (stride_y, stride_x) or single number if they're the same.
            dilations: spacing between kernel points (also known as astrous convolution),
                       either tuple (dilation_y, dilation_x) or single number if they're the same.
            padding: padding of the input tensor, either Padding.SAME, Padding.VALID or numerical values.
            use_bias: if True then convolution will have bias term.
            w_init: initializer for convolution kernel (a function that takes in a HWIO shape and returns a 4D matrix).
        """
        super().__init__(nin=nout, nout=nin, k=k, strides=strides, dilations=dilations, padding=padding,
                         use_bias=False, w_init=w_init)
        self.b = TrainVar(jn.zeros((nout, 1, 1), dtype=jn.float32)) if use_bias else None

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the transposed convolution to input x."""
        y = lax.conv_transpose(x, self.w.value, self.strides, self.padding,
                               rhs_dilation=self.dilations,
                               dimension_numbers=('NCHW', 'HWIO', 'NCHW'), transpose_kernel=True)
        if self.b is not None:
            y += self.b.value
        return y

    def __repr__(self):
        args = dict(nin=self.w.value.shape[3], nout=self.w.value.shape[2], k=self.w.value.shape[:2],
                    strides=self.strides, dilations=self.dilations, padding=self.padding,
                    use_bias=self.b is not None)
        args = ', '.join(f'{k}={repr(v)}' for k, v in args.items())
        return f'{class_name(self)}({args}, w_init={util.repr_function(self.w_init)})'

class Linear(objax.Module):
    """Applies a linear transformation on an input batch."""

    def __init__(self, nin: int, nout: int, use_bias: bool = True, w_init: Callable = xavier_normal):
        """Creates a Linear module instance.

        Args:
            nin: number of channels of the input tensor.
            nout: number of channels of the output tensor.
            use_bias: if True then linear layer will have bias term.
            w_init: weight initializer for linear layer (a function that takes in a IO shape and returns a 2D matrix).
        """
        super().__init__()
        self.w_init = w_init
        self.b = TrainVar(jn.zeros(nout, dtype=jn.float32)) if use_bias else None
        self.w = TrainVar(w_init((nin, nout)).astype(jn.float32))

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the linear transformation to input x."""
        y = jn.dot(x, self.w.value)
        if self.b is not None:
            y += self.b.value
        return y

    def __repr__(self):
        s = self.w.value.shape
        args = f'nin={s[0]}, nout={s[1]}, use_bias={self.b is not None}, w_init={util.repr_function(self.w_init)}'
        return f'{class_name(self)}({args})'