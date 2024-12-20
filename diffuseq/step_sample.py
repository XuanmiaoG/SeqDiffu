from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusions):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusions: the list of diffusion objects for multi-scale diffusion.
    """
    if name == "uniform":
        return MultiScaleSampler(UniformSampler, diffusions)
    elif name == "lossaware":
        return MultiScaleSampler(LossSecondMomentResampler, diffusions)
    elif name == "fixstep":
        return MultiScaleSampler(FixSampler, diffusions)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class FixSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.concatenate([np.ones([diffusion.num_timesteps // 2]),
                                        np.zeros([diffusion.num_timesteps // 2]) + 0.5])

    def weights(self):
        return self._weights


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class MultiScaleSampler(ScheduleSampler):
    """
    A sampler that handles multiple diffusion models for multi-scale diffusion.

    :param sampler_class: The base sampler class to use for each scale.
    :param diffusions: A list of diffusion objects for each scale.
    """

    def __init__(self, sampler_class, diffusions):
        self.samplers = [sampler_class(diffusion) for diffusion in diffusions]

    def weights(self):
        """
        Combine weights across all scales.
        """
        all_weights = [sampler.weights() for sampler in self.samplers]
        combined_weights = np.concatenate(all_weights)
        return combined_weights

    def sample(self, batch_size, device):
        """
        Sample timesteps from all scales.
        """
        total_timesteps = sum(sampler.diffusion.num_timesteps for sampler in self.samplers)
        p = np.concatenate([sampler.weights() for sampler in self.samplers])
        p /= np.sum(p)

        indices_np = np.random.choice(total_timesteps, size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (total_timesteps * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights
