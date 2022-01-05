import pickle
from tensorflow import keras
import numpy as np
import random
import os
import sys
import itertools
import copy
from collections import OrderedDict
from time import strftime, localtime
from helpers import schedulers
from helpers.data_generator import process_data, AutoEncoderDataGenerator
from helpers.hyperparam_helpers import make_bash_scripts
from helpers.custom_losses import (
    denorm_loss,
    hinge_mse_loss,
    percent_baseline_error,
    percent_correct_sign,
    baseline_MAE,
)
from helpers.custom_constraints import Orthonormal
import models.autoencoder
from helpers.callbacks import CyclicLR, TensorBoardWrapper, TimingCallback
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
import tensorflow as tf
from tensorflow.keras import backend as K


def main(scenario_index=-2):

    ###################
    # set session
    ###################
    num_cores = 8
    req_mem = 80  # gb
    ngpu = 1
    # seed_value= 0
    # os.environ['PYTHONHASHSEED']=str(seed_value)
    # random.seed(seed_value)
    # np.random.seed(seed_value)
    # tf.set_random_seed(seed_value)

    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=4 * num_cores,
        inter_op_parallelism_threads=4 * num_cores,
        allow_soft_placement=True,
        device_count={"CPU": 1, "GPU": ngpu},
    )
    session = tf.compat.v1.Session(config=config)

    ###############
    # global stuff
    ###############

    checkpt_dir = "/projects/EKOLEMEN/profile_predictor/LRAN_12_22_21/"
    if not os.path.exists(checkpt_dir):
        os.makedirs(checkpt_dir)

    ###############
    # scenarios
    ###############

    efit_type = "EFIT01"
    default_scenario = {
        "actuator_names": ["pinj", "tinj", "curr_target", "target_density", "bt"],
        "profile_names": [
            "temp",
            "dens",
            "rotation",
            "press_{}".format(efit_type),
            "q_{}".format(efit_type),
        ],
        "scalar_names": [
            "density_estimate",
            "curr",
            "a_{}".format(efit_type),
            "betan_{}".format(efit_type),
            "drsep_{}".format(efit_type),
            "kappa_{}".format(efit_type),
            "li_{}".format(efit_type),
            "rmagx_{}".format(efit_type),
            "zmagX_{}".format(efit_type),
            "volume_{}".format(efit_type),
            "triangularity_top_{}".format(efit_type),
            "triangularity_bot_{}".format(efit_type),
        ],
        "profile_downsample": 2,
        "state_encoder_type": "dense",
        "state_decoder_type": "dense",
        "control_encoder_type": "none",
        "control_decoder_type": "none",
        "state_encoder_kwargs": {
            "num_layers": 6,
            "layer_scale": 2,
            "activation": "elu",
            "norm": True,
        },
        "state_decoder_kwargs": {
            "num_layers": 6,
            "layer_scale": 2,
            "activation": "elu",
        },
        "control_encoder_kwargs": {},
        "control_decoder_kwargs": {},
        "state_latent_dim": 50,
        "control_latent_dim": 5,
        "x_weight": 1,
        "u_weight": 1,
        "discount_factor": 1,
        "batch_size": 64,
        "epochs": 200,
        "flattop_only": True,
        "raw_data_path": "/projects/EKOLEMEN/profile_predictor/DATA/profile_data_50ms.pkl",
        "process_data": True,
        "invert_q": True,
        "optimizer": "adam",
        "optimizer_kwargs": {"lr": 0.001},
        "shuffle_generators": True,
        "pruning_functions": [
            "remove_nan",
            "remove_dudtrip",
            "remove_outliers",
        ],
        "normalization_method": None,
        "window_length": 1,
        "window_overlap": 0,
        "lookahead": 6,
        "sample_step": 1,
        "uniform_normalization": True,
        "train_frac": 0.8,
        "val_idx": np.random.randint(1, 10),
        "val_frac": 0.2,
        "nshots": 12000,
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

    IC_coils = [
        "C_coil_139",
        "C_coil_19",
        "C_coil_199",
        "C_coil_259",
        "C_coil_319",
        "C_coil_79",
        "I_coil_150L",
        "I_coil_150U",
        "I_coil_210L",
        "I_coil_210U",
        "I_coil_270L",
        "I_coil_270U",
        "I_coil_30L",
        "I_coil_30U",
        "I_coil_330L",
        "I_coil_330U",
        "I_coil_90L",
        "I_coil_90U",
    ]
    F_coils = [
        "F_coil_1a",
        "F_coil_1b",
        "F_coil_2a",
        "F_coil_2b",
        "F_coil_3a",
        "F_coil_3b",
        "F_coil_4a",
        "F_coil_4b",
        "F_coil_5a",
        "F_coil_5b",
        "F_coil_6a",
        "F_coil_6b",
        "F_coil_7a",
        "F_coil_7b",
        "F_coil_8a",
        "F_coil_8b",
        "F_coil_9a",
        "F_coil_9b",
    ]

    scenarios_dict = OrderedDict()
    scenarios_dict["actuator_names"] = [
        {
            "actuator_names": [
                "pinj_30L",
                "pinj_30R",
                "pinj_15L",
                "pinj_15R",
                "pinj_21L",
                "pinj_21R",
                "pinj_33L",
                "pinj_33R",
                "curr_target",
                "target_density",
                "bt",
                "ech",
            ]
        },
        {
            "actuator_names": [
                "pinj",
                "tinj",
                "curr_target",
                "target_density",
                "bt",
                "ech",
            ]
        },
        {
            "actuator_names": [
                "pinj_30L",
                "pinj_30R",
                "pinj_15L",
                "pinj_15R",
                "pinj_21L",
                "pinj_21R",
                "pinj_33L",
                "pinj_33R",
                "curr_target",
                "target_density",
                "bt",
            ]
            + IC_coils
        },
        {
            "actuator_names": [
                "pinj",
                "tinj",
                "curr_target",
                "target_density",
                "bt",
            ]
            + IC_coils
        },
        {
            "actuator_names": [
                "pinj_30L",
                "pinj_30R",
                "pinj_15L",
                "pinj_15R",
                "pinj_21L",
                "pinj_21R",
                "pinj_33L",
                "pinj_33R",
                "curr_target",
                "target_density",
                "bt",
                "ech",
            ]
            + IC_coils
        },
        {
            "actuator_names": [
                "pinj",
                "tinj",
                "curr_target",
                "target_density",
                "bt",
                "ech",
            ]
            + IC_coils
        },
    ]
    scenarios_dict["flattop_only"] = [{"flattop_only": True}, {"flattop_only": False}]
    scenarios_dict["state_latent_dim"] = [
        {"state_latent_dim": 165},
        {"state_latent_dim": 177},
        {"state_latent_dim": 200},
    ]
    scenarios_dict["lookahead"] = [
        {"lookahead": 20},
    ]
    scenarios_dict["state_encoder_kwargs"] = [
        {
            "state_encoder_kwargs": {
                "num_layers": 4,
                "layer_scale": 1,
                "activation": "elu",
                "norm": True,
            },
            "state_decoder_kwargs": {
                "num_layers": 4,
                "layer_scale": 1,
                "activation": "elu",
            },
        },
        {
            "state_encoder_kwargs": {
                "num_layers": 4,
                "layer_scale": 1,
                "activation": "leaky_relu",
                "norm": True,
            },
            "state_decoder_kwargs": {
                "num_layers": 4,
                "layer_scale": 1,
                "activation": "leaky_relu",
            },
        },
        {
            "state_encoder_kwargs": {
                "num_layers": 6,
                "layer_scale": 1,
                "activation": "elu",
                "norm": True,
            },
            "state_decoder_kwargs": {
                "num_layers": 6,
                "layer_scale": 1,
                "activation": "elu",
            },
        },
        {
            "state_encoder_kwargs": {
                "num_layers": 6,
                "layer_scale": 1,
                "activation": "leaky_relu",
                "norm": True,
            },
            "state_decoder_kwargs": {
                "num_layers": 6,
                "layer_scale": 1,
                "activation": "leaky_relu",
            },
        },
        {
            "state_encoder_kwargs": {
                "num_layers": 8,
                "layer_scale": 1,
                "activation": "elu",
                "norm": True,
            },
            "state_decoder_kwargs": {
                "num_layers": 8,
                "layer_scale": 1,
                "activation": "elu",
            },
        },
        {
            "state_encoder_kwargs": {
                "num_layers": 8,
                "layer_scale": 1,
                "activation": "leaky_relu",
                "norm": True,
            },
            "state_decoder_kwargs": {
                "num_layers": 8,
                "layer_scale": 1,
                "activation": "leaky_relu",
            },
        },
        {
            "state_encoder_kwargs": {
                "num_layers": 10,
                "layer_scale": 1,
                "activation": "elu",
                "norm": True,
            },
            "state_decoder_kwargs": {
                "num_layers": 10,
                "layer_scale": 1,
                "activation": "elu",
            },
        },
        {
            "state_encoder_kwargs": {
                "num_layers": 10,
                "layer_scale": 1,
                "activation": "leaky_relu",
                "norm": True,
            },
            "state_decoder_kwargs": {
                "num_layers": 10,
                "layer_scale": 1,
                "activation": "leaky_relu",
            },
        },
    ]

    scenarios = []
    runtimes = []
    for scenario in itertools.product(*list(scenarios_dict.values())):
        foo = {k: v for d in scenario for k, v in d.items()}
        scenarios.append(foo)
        runtimes.append(6 * 60)
    num_scenarios = len(scenarios)

    ###############
    # Batch Run
    ###############
    if scenario_index == -1:
        make_bash_scripts(
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

    if (scenario["control_encoder_type"] == "none") or (
        scenario["control_decoder_type"] == "none"
    ):
        assert scenario["control_encoder_type"] == "none"
        assert scenario["control_decoder_type"] == "none"
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

    scenario["dt"] = np.mean(np.diff(traindata["time"])) / 1000  # in seconds
    scenario["normalization_dict"] = normalization_dict

    scenario["profile_length"] = int(np.ceil(65 / scenario["profile_downsample"]))

    scenario["runname"] = "LRAN" + strftime("_%d%b%y-%H-%M", localtime())
    if scenario_index >= 0:
        scenario["runname"] += "_Scenario-{:04d}".format(scenario_index)
    scenario["model_path"] = checkpt_dir + scenario["runname"] + "_model.h5"

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
    )

    print("Made Generators")

    ###############
    # Get model and optimizer
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

    model = models.autoencoder.make_autoencoder(
        scenario["state_encoder_type"],
        scenario["state_decoder_type"],
        scenario["control_encoder_type"],
        scenario["control_decoder_type"],
        scenario["state_encoder_kwargs"],
        scenario["state_decoder_kwargs"],
        scenario["control_encoder_kwargs"],
        scenario["control_decoder_kwargs"],
        scenario["profile_names"],
        scenario["scalar_names"],
        scenario["actuator_names"],
        scenario["state_latent_dim"],
        scenario["control_latent_dim"],
        scenario["profile_length"],
        scenario["lookahead"],
        None,  # scenario["batch_size"],
    )

    model.summary()

    optimizer = optimizers[scenario["optimizer"]](**scenario["optimizer_kwargs"])

    ###############
    # Get losses and metrics
    ###############

    loss = "mse"
    metrics = ["mse", "mae"]
    callbacks = []
    schedules = {
        "exp": schedulers.exp,
        "poly": schedulers.poly,
        "piece": schedulers.piece,
        "inverseT": schedulers.decayed_learning_rate,
    }
    if "lr_schedule" in scenario:
        schedule = schedules[scenario["lr_schedule"]](**scenario.get("lr_kwargs", {}))
        callbacks.append(LearningRateScheduler(schedule=schedule, verbose=1))
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=50,
            verbose=1,
            mode="auto",
            min_delta=0.001,
            cooldown=1,
            min_lr=0,
        )
    )
    callbacks.append(
        EarlyStopping(
            monitor="val_loss", min_delta=0, patience=50, verbose=1, mode="min"
        )
    )
    callbacks.append(TimingCallback(time_limit=(runtimes[scenario_index] - 30) * 60))
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
    # Compile and Train
    ###############
    model.compile(optimizer, loss, metrics, sample_weight_mode="temporal")
    print("Model compiled, starting training")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=scenario["steps_per_epoch"],
        epochs=scenario["epochs"],
        callbacks=callbacks,
        validation_data=val_generator,
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
