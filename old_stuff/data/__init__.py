"""
Keras dataset specifications.
TODO: add MNIST.
"""

def get_datasets(name, **data_args):
    if name == 'dummy':
        from .dummy import get_datasets
        return get_datasets(**data_args)
    elif name == 'cifar10':
        from .cifar10 import get_datasets
        return get_datasets(**data_args)
    elif name == 'imagenet':
        from .imagenet import get_datasets
        return get_datasets(**data_args)
    elif name == 'rnn':
        from .rnn import get_datasets
        return get_datasets(**data_args)
    elif name == 'lstm_cnn':
        from .lstm_cnn import get_datasets
        return get_datasets(**data_args)
    elif name == 'trend_plus_actuators':
        from .trend_plus_actuators import get_datasets
        return get_datasets(**data_args)
    else:
        raise ValueError('Dataset %s unknown' % name)
