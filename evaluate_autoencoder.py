import sys
import os
import pickle
from tensorflow import keras
import datetime
import matplotlib
import copy
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./"))
import helpers.mpc_helpers
from helpers.data_generator import process_data, AutoEncoderDataGenerator
from helpers.normalization import normalize, denormalize, renormalize
from helpers.custom_layers import MultiTimeDistributed


base_path = "/projects/EKOLEMEN/profile_predictor/"
folders = ["LRAN_11_30_21/"]


##########
# set tf session
##########
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=16,
    inter_op_parallelism_threads=16,
    allow_soft_placement=True,
    device_count={"CPU": 8, "GPU": 1},
)
session = tf.compat.v1.Session(config=config)

##########
# metrics
##########


def sigma(inp, true, prediction):
    eps = prediction - true

    num = np.linalg.norm(eps, axis=-1)
    denom = np.linalg.norm(true, axis=-1)

    included_inds = np.where(~np.isclose(denom, 0))[0]
    return num[included_inds] / denom[included_inds]


def mean_squared_error(residual):
    return np.mean((residual) ** 2)


def mean_absolute_error(residual):
    return np.mean(np.abs(residual))


def median_absolute_error(residual):
    return np.median(np.abs(residual))


def percentile25_absolute_error(residual):
    return np.percentile(np.abs(residual), 25)


def percentile75_absolute_error(residual):
    return np.percentile(np.abs(residual), 75)


def median_squared_error(residual):
    return np.median((residual) ** 2)


def percentile25_squared_error(residual):
    return np.percentile((residual) ** 2, 25)


def percentile75_squared_error(residual):
    return np.percentile((residual) ** 2, 75)


metrics = {
    "mean_squared_error": mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
    "median_absolute_error": median_absolute_error,
    "percentile25_absolute_error": percentile25_absolute_error,
    "percentile75_absolute_error": percentile75_absolute_error,
    "median_squared_error": median_squared_error,
    "percentile25_squared_error": percentile25_squared_error,
    "percentile75_squared_error": percentile75_squared_error,
    # "sigma": sigma,
}

##########
# load model and scenario
##########

for folder in folders:
    files = [
        os.path.join(base_path, folder, foo)
        for foo in os.listdir(os.path.join(base_path, folder))
        if foo.endswith(".pkl")
    ]

    for file_path in files:
        try:
            print("loading scenario: " + file_path)
            with open(file_path, "rb") as f:
                scenario = pickle.load(f, encoding="latin1")

            model_path = file_path[:-11] + "_model.h5"
            prev_time = time.time()
            if os.path.exists(model_path):
                print("loading model: " + model_path.split("/")[-1])
                model = keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects={"MultiTimeDistributed": MultiTimeDistributed},
                )
                print("took {}s".format(time.time() - prev_time))
            else:
                print("no model for path:", model_path)
                continue

            prev_time = time.time()

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
                0,  # verbose,
                scenario["flattop_only"],
                pruning_functions=scenario["pruning_functions"],
                invert_q=scenario["invert_q"],
                val_idx=0,
                excluded_shots=scenario["excluded_shots"],
            )
            print("Data processing took {}s".format(time.time() - prev_time))

            val_generator = AutoEncoderDataGenerator(
                valdata,
                scenario["batch_size"],
                scenario["profile_names"],
                scenario["actuator_names"],
                scenario["scalar_names"],
                scenario["lookahead"],
                scenario["profile_downsample"],
                scenario["state_latent_dim"],
                1,  # scenario["discount_factor"],
                1,  # scenario["x_weight"],
                1,  # scenario["u_weight"],
                False,
                sample_weights=None,
            )

            ures, xres, lres = model.predict(
                val_generator, verbose=0, workers=4, use_multiprocessing=True
            )
            x_residuals = {
                sig: xres[..., i * 33 : (i + 1) * 33]
                for i, sig in enumerate(scenario["profile_names"])
            }

            evaluation_metrics = {}
            for metric_name, metric in metrics.items():
                s = 0
                key = "linear_sys_" + metric_name
                val = metric(lres)
                print(key)
                print(val)
                evaluation_metrics[key] = val
                for sig in scenario["profile_names"]:
                    key = sig + "_" + metric_name
                    val = metric(x_residuals[sig])
                    s += val / len(scenario["profile_names"])
                    evaluation_metrics[key] = val
                    print(key)
                    print(val)
                evaluation_metrics[metric_name] = s

            encoder_data = helpers.mpc_helpers.compute_encoder_data(
                model, scenario, scenario["raw_data_path"], verbose=0
            )
            norm_data = helpers.mpc_helpers.compute_norm_data(
                encoder_data["x0"], encoder_data["z0"]
            )

            evaluation_metrics["median_operator_norm"] = np.nanmedian(
                norm_data["operator_norm"]
            )
            evaluation_metrics["median_lipschitz_const"] = np.nanmedian(
                norm_data["lipschitz_constant"]
            )
            evaluation_metrics["std_operator_norm"] = np.nanstd(
                norm_data["operator_norm"]
            )
            evaluation_metrics["std_lipschitz_const"] = np.nanstd(
                norm_data["lipschitz_constant"]
            )

            scenario["norm_data"] = norm_data
            scenario["evaluation_metrics"] = evaluation_metrics

            prev_time = time.time()
            with open(file_path, "wb+") as f:
                pickle.dump(copy.deepcopy(scenario), f)
            print("Repickling took {}s".format(time.time() - prev_time))

            print("saved evaluation metrics")
        except Exception as e:
            print(e)

    print("done")
