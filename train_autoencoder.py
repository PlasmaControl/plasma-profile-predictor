import os
import sys
import copy
import time
import pickle
import random
import itertools
import collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
import models.autoencoder
import helpers
from helpers import schedulers
from helpers import signal_groups
from helpers.data_generator import process_data, AutoEncoderDataGenerator
from helpers.callbacks import TimingCallback


def main(scenario_index=-2):

    ###################
    # set session
    ###################
    num_cores = 8
    req_mem = 80  # gb
    ngpu = 1

    seed_value = 0
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    ###############
    # global stuff
    ###############

    checkpt_dir = "/projects/EKOLEMEN/profile_predictor/LRAN_01_24_22/"
    if not os.path.exists(checkpt_dir):
        os.makedirs(checkpt_dir)

    ###############
    # Default set of hyperparameters
    ###############

    efit_type = "EFIT01"
    default_scenario = {
        ### names of signals to use
        "actuator_names": ["pinj", "tinj", "curr_target", "target_density"],
        "profile_names": [
            "temp",
            "dens",
            "rotation",
            "press_{}".format(efit_type),
            "q_{}".format(efit_type),
        ],
        "scalar_names": [
            # "density_estimate",
            # "curr",
            # "a_{}".format(efit_type),
            # "betan_{}".format(efit_type),
            # "drsep_{}".format(efit_type),
            # "kappa_{}".format(efit_type),
            # "li_{}".format(efit_type),
            # "rmagx_{}".format(efit_type),
            # "zmagX_{}".format(efit_type),
            # "volume_{}".format(efit_type),
            # "triangularity_top_{}".format(efit_type),
            # "triangularity_bot_{}".format(efit_type),
        ],
        ### what type of model to use and settings etc.
        "state_encoder_type": "dense",
        "control_encoder_type": "none",
        "state_encoder_kwargs": {
            "num_layers": 6,
            "activation": "elu",
            "norm": True,
        },
        "control_encoder_kwargs": {},
        ### dimension of latent states: negative means same as physical state, scaled
        "state_latent_dim": -1,
        "control_latent_dim": -1,
        ### weighting for different terms. latent state loss == 1
        "sample_weights": True,  # True to weight samples temporally, "std" to weight by how much things are changing
        "x_weight": 1,  # state encode/decode error
        "u_weight": 1,  # control encode/decode error
        "discount_factor": 1,  # reduction ratio for future predictions
        ### loss and training parameters
        "optimizer": "adam",
        "optimizer_kwargs": {"lr": 0.001},
        "shuffle_generators": True,  # re-order samples on each epoch
        "loss_function": "mse",
        "loss_function_kwargs": {},
        "batch_size": 64,
        "epochs": 400,
        ### data processing stuff
        "raw_data_path": "/projects/EKOLEMEN/profile_predictor/DATA/profile_data_50ms.pkl",
        "flattop_only": True,  # only include data during "steady state"
        "invert_q": True,  # to avoid singularity at psi=1
        "normalization_method": "RobustScaler",  # if normalizing data beforehand, None if using BatchNormalization layers
        "uniform_normalization": True,  # whether to use same mean/std for entire profile
        "profile_downsample": 2,  # by default. profiles are 65 pts long, so may be downsampled
        "lookahead": 10,  # horizon for prediction / control, in 50ms chunks
        ### leave these params alone
        "lookback": 0,  # should always be 0 state space model
        "window_length": 1,  # number of samples to average over
        "window_overlap": 0,  # how much to overlap averaging windows
        "sample_step": 1,  # number of samples between starts of sequential samples
        ### how to split up training/validation
        "train_frac": 1,  # fraction in (0,1)
        "val_frac": 0,  # fraction in (0,1)
        "val_idx": np.random.randint(
            1, 10
        ),  # override frac, use specific shots instead
        "nshots": np.inf,  # maximum number of shots to use
        ### what data to exclude from training
        "pruning_functions": [
            "remove_nan",  # remove junk data
            "remove_dudtrip",  # remove PCS crashes etc
            "remove_outliers",  # remove weird fits and bad postprocessing
        ],
        ### lists of specific shots to exclude, mostly based on rare topology
        "excluded_shots": [
            "topology_TOP",
            "topology_OUT",
            "topology_MAR",
            "topology_IN",
            "topology_DN",
            "topology_BOT",
            "test_set",
        ],
    }

    ###############
    # For hyperparameter scans
    ###############
    scenarios_dict = collections.OrderedDict()
    scenarios_dict["actuator_names"] = [
        {
            "actuator_names": [
                "pinj",
                "tinj",
                "curr_target",
                "target_density",
            ]
        },
    ]
    scenarios_dict["flattop_only"] = [{"flattop_only": False}]
    scenarios_dict["sample_weights"] = [
        {"sample_weights": "std"},
        {"sample_weights": True},
    ]
    scenarios_dict["state_latent_dim"] = [
        {"state_latent_dim": 50},
        {"state_latent_dim": 75},
    ]
    scenarios_dict["lookahead"] = [
        {"lookahead": 20},
    ]
    scenarios_dict["loss"] = [
        {"loss_function": "mse"},
        {"loss_function": "mae"},
        {"loss_function": "logcosh"},
    ]
    scenarios_dict["state_encoder_kwargs"] = [
        {
            "state_encoder_kwargs": {
                "num_layers": 4,
                "activation": "leaky_relu",
                "norm": True,
                "layer_scale": np.inf,
            },
        },
        {
            "state_encoder_kwargs": {
                "num_layers": 6,
                "activation": "leaky_relu",
                "norm": True,
                "layer_scale": np.inf,
            },
        },
        {
            "state_encoder_kwargs": {
                "num_layers": 8,
                "activation": "leaky_relu",
                "norm": True,
                "layer_scale": np.inf,
            },
        },
    ]

    scenarios = []
    runtimes = []
    for scenario in itertools.product(*list(scenarios_dict.values())):
        foo = {k: v for d in scenario for k, v in d.items()}
        scenarios.append(foo)
        runtimes.append(18 * 60)
    num_scenarios = len(scenarios)

    ###############
    # Batch Run
    ###############
    if scenario_index == -1:
        helpers.hyperparam_helpers.make_bash_scripts(
            num_scenarios,
            checkpt_dir,
            num_cores,
            ngpu,
            req_mem,
            runtimes,
            mode="autoencoder",
        )
        print("Created Driver Scripts in " + checkpt_dir)
        print("Jobs submitted, exiting")
        return

    ###############
    # Load Scenario and Data
    ###############
    if scenario_index >= 0:
        verbose = 2
        print("Loading Scenario " + str(scenario_index) + ":")
        scenario = scenarios[scenario_index]
        scenario.update(
            {k: v for k, v in default_scenario.items() if k not in scenario.keys()}
        )
    else:
        verbose = 2
        print("Loading Default Scenario:")
        scenario = default_scenario

    if (scenario["control_encoder_type"] == "none") or ():
        assert scenario["control_encoder_type"] == "none"
        scenario["control_latent_dim"] = len(scenario["actuator_names"])

    if scenario["control_latent_dim"] < 0:
        scenario["control_latent_dim"] = int(
            abs(scenario["control_latent_dim"]) * len(scenario["actuator_names"])
        )
    if scenario["state_latent_dim"] < 0:
        scenario["state_latent_dim"] = int(
            abs(scenario["state_latent_dim"])
            * (
                np.ceil(65 / scenario["profile_downsample"])
                * len(scenario["profile_names"])
                + len(scenario["scalar_names"])
            )
        )

    for k, v in scenario.items():
        print("{}:{}".format(k, v))

    scenario["sig_names"] = (
        scenario["profile_names"]
        + scenario["actuator_names"]
        + scenario["scalar_names"]
    )

    traindata, valdata, normalization_dict = process_data(
        scenario["raw_data_path"],
        scenario["sig_names"],
        scenario["normalization_method"],
        scenario["window_length"],
        scenario["window_overlap"],
        0,  # scenario['lookback'],
        scenario["lookahead"],
        scenario["sample_step"],
        scenario["uniform_normalization"],
        scenario["train_frac"],
        scenario["val_frac"],
        scenario["nshots"],
        verbose,
        scenario["flattop_only"],
        pruning_functions=scenario["pruning_functions"],
        invert_q=scenario["invert_q"],
        val_idx=scenario["val_idx"],
        excluded_shots=scenario["excluded_shots"],
    )
    scenario["normalization_dict"] = normalization_dict
    scenario["dt"] = np.mean(np.diff(traindata["time"])) / 1000  # in seconds
    scenario["profile_length"] = int(np.ceil(65 / scenario["profile_downsample"]))

    scenario["runname"] = "LRAN" + time.strftime("_%d%b%y-%H-%M", time.localtime())
    if scenario_index >= 0:
        scenario["runname"] += "_Scenario-{:04d}".format(scenario_index)
    scenario["model_path"] = checkpt_dir + scenario["runname"] + "_model.tf"
    print(scenario["runname"])

    ###############
    # Make data generators
    ###############

    train_generator = AutoEncoderDataGenerator(
        traindata,
        scenario["batch_size"],
        scenario["profile_names"],
        scenario["actuator_names"],
        scenario["scalar_names"],
        scenario["lookahead"],
        scenario["profile_downsample"],
        scenario["state_latent_dim"],
        scenario["discount_factor"],
        scenario["x_weight"],
        scenario["u_weight"],
        scenario["shuffle_generators"],
        sample_weights=scenario["sample_weights"],
    )
    val_generator = AutoEncoderDataGenerator(
        valdata,
        scenario["batch_size"],
        scenario["profile_names"],
        scenario["actuator_names"],
        scenario["scalar_names"],
        scenario["lookahead"],
        scenario["profile_downsample"],
        scenario["state_latent_dim"],
        scenario["discount_factor"],
        scenario["x_weight"],
        scenario["u_weight"],
        scenario["shuffle_generators"],
        sample_weights=scenario["sample_weights"],
    )
    print("Made Generators")

    ###############
    # Get optimizer, losses metrics, callbacks
    ###############
    optimizers = {
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
        "adagrad": keras.optimizers.Adagrad,
        "adadelta": keras.optimizers.Adadelta,
        "adam": keras.optimizers.Adam,
        "adamax": keras.optimizers.Adamax,
        "nadam": keras.optimizers.Nadam,
    }
    optimizer = optimizers[scenario["optimizer"]](**scenario["optimizer_kwargs"])

    loss = scenario["loss_function"]

    metrics = ["mse", "mae", "logcosh"]

    callbacks = []
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.25,
            patience=10,
            verbose=1,
            mode="min",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        )
    )
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=12,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )
    )
    callbacks.append(
        TimingCallback(
            time_limit=(runtimes[scenario_index] - 30) * 60,
        )
    )
    callbacks.append(
        ModelCheckpoint(
            scenario["model_path"],
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1,
        )
    )
    schedules = {
        "exp": schedulers.exp,
        "poly": schedulers.poly,
        "piece": schedulers.piece,
        "inverseT": schedulers.decayed_learning_rate,
    }
    if "lr_schedule" in scenario:
        schedule = schedules[scenario["lr_schedule"]](**scenario.get("lr_kwargs", {}))
        callbacks.append(LearningRateScheduler(schedule=schedule, verbose=1))

    scenario["steps_per_epoch"] = len(train_generator)
    scenario["val_steps"] = len(val_generator)
    print("Train generator length: {}".format(len(train_generator)))

    ###############
    # Save scenario
    ###############
    with open(checkpt_dir + scenario["runname"] + "_params.pkl", "wb+") as f:
        pickle.dump(copy.deepcopy(scenario), f)
    print("Saved Analysis params before run")

    ###############
    # Get and compile model
    ###############
    model = models.autoencoder.make_autoencoder(
        scenario["state_encoder_type"],
        scenario["control_encoder_type"],
        scenario["state_encoder_kwargs"],
        scenario["control_encoder_kwargs"],
        scenario["recurrent_kwargs"],
        scenario["profile_names"],
        scenario["scalar_names"],
        scenario["actuator_names"],
        scenario["state_latent_dim"],
        scenario["control_latent_dim"],
        scenario["profile_length"],
        scenario["lookahead"],
    )
    model.summary()
    model.compile(optimizer, loss, metrics, sample_weight_mode="temporal")
    print("Model compiled, starting training")

    ###############
    # Do the training thing
    ###############
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        callbacks=callbacks,
        epochs=scenario["epochs"],
        steps_per_epoch=scenario["steps_per_epoch"],
        validation_steps=scenario["val_steps"],
        verbose=verbose,
    )

    ###############
    # Save Results
    ###############
    scenario["history"] = history.history
    scenario["history_params"] = history.params

    if not any([isinstance(cb, ModelCheckpoint) for cb in callbacks]):
        model.save(scenario["model_path"])
        print("Saved model after training")
    with open(checkpt_dir + scenario["runname"] + "_params.pkl", "wb+") as f:
        pickle.dump(copy.deepcopy(scenario), f)
    print("Saved Analysis params after training")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()
