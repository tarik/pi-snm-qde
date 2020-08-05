import torch

__all__ = [
    'Adam',
    'SGD',
    'ExponentialDecay'
]


Adam = torch.optim.Adam
SGD = torch.optim.SGD


class ExponentialDecay(torch.optim.lr_scheduler.LambdaLR):
    """
    See: https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay.

    `lr = initial_lr * decay_rate ^ (step / decay_steps)`
    """

    def __init__(self, optimizer, decay_rate, decay_steps, **kwargs):
        def lr_lambda(step):
            return decay_rate ** (step / decay_steps)
        super().__init__(optimizer, lr_lambda=lr_lambda)
