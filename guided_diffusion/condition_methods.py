from abc import ABC, abstractmethod
import torch
from torch.autograd.functional import jacobian
from util.distance_utils import mapping_base_on_hypersphere_constraint

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
        # self.sample = sampler

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t


@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale

        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm, torch.zeros(1)


@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        print(f'dps_scale:{self.scale}')

    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        r = torch.linalg.norm(x_t - x_t_mean, dim=[1, 2, 3])  # (bs)
        return x_t, norm

# Implement of Diffusion with Spherical Gaussian Constraint(DSG)
@register_conditioning_method(name='DSG')
class DSG(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.interval = kwargs.get('interval', 1.)
        self.guidance_scale = kwargs.get('guidance_scale', 1.)
        print(f'interval: {self.interval}')
        print(f'guidance_scale: {self.guidance_scale}')

    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, idx, **kwargs):
        eps = 1e-8
        if idx % self.interval == 0:
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
            grad_norm = torch.linalg.norm(grad, dim=[1, 2, 3])

            b, c, h, w = x_t.shape
            r = torch.sqrt(torch.tensor(c * h * w)) * kwargs.get('sigma_t', 1.)[0, 0, 0, 0]
            guidance_rate = self.guidance_scale

            d_star = -r * grad / (grad_norm + eps)
            d_sample = x_t - x_t_mean
            mix_direction = d_sample + guidance_rate * (d_star - d_sample)
            mix_direction_norm = torch.linalg.norm(mix_direction, dim=[1, 2, 3])
            mix_step = mix_direction / (mix_direction_norm + eps) * r

            return x_t_mean + mix_step, norm

        else:
            # use the code below for print loss in unconditional step and commit it for saving time
            # difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            # norm = torch.linalg.norm(difference)
            # return x_t, norm

            return x_t, torch.zeros(1)
