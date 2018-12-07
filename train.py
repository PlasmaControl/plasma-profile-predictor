"""
Main training script for the Deep Learning at Scale Keras examples.
"""

# System
import os
import sys
import argparse
import logging

# Externals
import keras
import horovod.keras as hvd
import yaml
import numpy as np

# Locals
from data import get_datasets
from models import get_model
from utils.device import configure_session
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-d', '--distributed', action='store_true')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--gpu', type=int,
            help='specify a gpu device ID if not running distributed')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def config_logging(verbose, output_dir):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'out.log'), mode='w')
    file_handler.setLevel(log_level)
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[stream_handler, file_handler])

def init_workers(distributed=False):
    rank, n_ranks = 0, 1
    if distributed:
        hvd.init()
        rank, n_ranks = hvd.rank(), hvd.size()
    return rank, n_ranks

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f)
    return config

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    rank, n_ranks = init_workers(args.distributed)

    # Load configuration
    config = load_config(args.config)
    train_config = config['training']
    output_dir = os.path.expandvars(config['output_dir'])
    checkpoint_format = os.path.join(output_dir, 'checkpoints',
                                     'checkpoint-{epoch}.h5')
    os.makedirs(output_dir, exist_ok=True)

    # Loggging
    config_logging(verbose=args.verbose, output_dir=output_dir)
    logging.info('Initialized rank %i out of %i', rank, n_ranks)
    if args.show_config:
        logging.info('Command line config: %s', args)
    if rank == 0:
        logging.info('Job configuration: %s', config)
        logging.info('Saving job outputs to %s', output_dir)

    # Configure session
    if args.distributed:
        gpu = hvd.local_rank()
    else:
        gpu = args.gpu
    device_config = config.get('device', {})
    configure_session(gpu=gpu, **device_config)

    # Load the data
    train_gen, valid_gen = get_datasets(batch_size=train_config['batch_size'],
                                        **config['data'])

    # Build the model
    model = get_model(**config['model'])
    # Configure optimizer
    opt = get_optimizer(n_ranks=n_ranks, distributed=args.distributed,
                        **config['optimizer'])
    # Compile the model
    model.compile(loss=train_config['loss'], optimizer=opt,
                  metrics=train_config['metrics'])
    if rank == 0:
        model.summary()

    # Prepare the training callbacks
    callbacks = []
    if args.distributed:

        # Broadcast initial variable states from rank 0 to all processes.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

        # Learning rate warmup
        warmup_epochs = train_config.get('lr_warmup_epochs', 0)
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(
            warmup_epochs=warmup_epochs, verbose=1))

        # Learning rate decay schedule
        for lr_schedule in train_config.get('lr_schedule', []):
            if rank == 0:
                logging.info('Adding LR schedule: %s', lr_schedule)
            callbacks.append(hvd.callbacks.LearningRateScheduleCallback(**lr_schedule))

    # Checkpoint only from rank 0
    if rank == 0:
        os.makedirs(os.path.dirname(checkpoint_format), exist_ok=True)
        callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_format))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Train the model
    steps_per_epoch = train_config['steps_per_epoch'] // n_ranks #len(train_gen) // n_ranks
    history = model.fit_generator(train_gen,
                                  epochs=train_config['n_epochs'],
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=valid_gen,
                                  validation_steps=train_config['validation_steps'], #len(valid_gen),
                                  callbacks=callbacks,
                                  workers=4, verbose=2)

    # Save training history
    if rank == 0:
        # Print some best-found metrics
        if 'val_acc' in history.history.keys():
            logging.info('Best validation accuracy: %.3f',
                         max(history.history['val_acc']))
        if 'val_top_k_categorical_accuracy' in history.history.keys():
            logging.info('Best top-5 validation accuracy: %.3f',
                         max(history.history['val_top_k_categorical_accuracy']))
        logging.info('Average time per epoch: %.3f s',
                     np.mean(timing_callback.times))
        np.savez(os.path.join(output_dir, 'history'),
                 n_ranks=n_ranks, **history.history)

    # Drop to IPython interactive shell
    if args.interactive and (rank == 0):
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    if rank == 0:
        logging.info('All done!')

if __name__ == '__main__':
    main()
