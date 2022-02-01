import sys
import os
import gc
import pickle
import datetime
import copy
import time

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./"))
import helpers.lran_helpers
from helpers.data_generator import process_data, AutoEncoderDataGenerator
from helpers.normalization import normalize, denormalize, renormalize
from helpers.hyperparam_helpers import slurm_script
from helpers import custom_objects

##########
# set tf session
##########
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=16,
    inter_op_parallelism_threads=16,
    allow_soft_placement=True,
    device_count={"CPU": len(os.sched_getaffinity(0)), "GPU": 0},
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


def mean_absolute_error(residual):
    return np.mean(np.linalg.norm(residual, ord=1, axis=-1))


def median_absolute_error(residual):
    return np.median(np.linalg.norm(residual, ord=1, axis=-1))


def percentile25_absolute_error(residual):
    return np.percentile(np.linalg.norm(residual, ord=1, axis=-1), 25)


def percentile75_absolute_error(residual):
    return np.percentile(np.linalg.norm(residual, ord=1, axis=-1), 75)


def percentile99_absolute_error(residual):
    return np.percentile(np.linalg.norm(residual, ord=1, axis=-1), 99)


def max_absolute_error(residual):
    return np.max(np.linalg.norm(residual, ord=1, axis=-1))


def root_mean_squared_error(residual):
    return np.mean(np.linalg.norm(residual, ord=2, axis=-1))


def root_median_squared_error(residual):
    return np.median(np.linalg.norm(residual, ord=2, axis=-1))


def root_percentile25_squared_error(residual):
    return np.percentile(np.linalg.norm(residual, ord=2, axis=-1), 25)


def root_percentile75_squared_error(residual):
    return np.percentile(np.linalg.norm(residual, ord=2, axis=-1), 75)


def root_percentile99_squared_error(residual):
    return np.percentile(np.linalg.norm(residual, ord=2, axis=-1), 99)


def root_max_squared_error(residual):
    return np.max(np.linalg.norm(residual, ord=2, axis=-1))


metrics = {
    "mean_absolute_error": mean_absolute_error,
    "median_absolute_error": median_absolute_error,
    "percentile25_absolute_error": percentile25_absolute_error,
    "percentile75_absolute_error": percentile75_absolute_error,
    "percentile99_absolute_error": percentile99_absolute_error,
    "max_absolute_error": max_absolute_error,
    "root_mean_squared_error": root_mean_squared_error,
    "root_median_squared_error": root_median_squared_error,
    "root_percentile25_squared_error": root_percentile25_squared_error,
    "root_percentile75_squared_error": root_percentile75_squared_error,
    "root_percentile99_squared_error": root_percentile99_squared_error,
    "root_max_squared_error": root_max_squared_error,
    # "sigma": sigma,
}


def evaluate(file_path):
    """Evaluate model on consistent set of data

    Parameters
    ----------
    file_path : str
        path to pkl file of scenario
    """

    T0 = time.time()
    print("loading scenario: " + file_path)
    with open(file_path, "rb") as f:
        scenario = pickle.load(f, encoding="latin1")

    T1 = time.time()
    print("took {}s".format(T1 - T0))

    fmt = scenario["model_path"][-2:]
    model_path = file_path[:-11] + "_model." + fmt
    if os.path.exists(model_path):
        print("loading model: " + model_path.split("/")[-1])
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects if (fmt == "h5") else None,
        )
        print("took {}s".format(time.time() - T1))
    else:
        print("no model for path:", model_path)
        return

    T1 = time.time()
    print("loading data")
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
        1,  # scenario['train_frac'],
        0,  # scenario['val_frac'],
        scenario["nshots"],
        2,  # verbose,
        scenario["flattop_only"],
        pruning_functions=scenario["pruning_functions"],
        invert_q=scenario["invert_q"],
        excluded_shots=scenario["excluded_shots"],
        val_idx=0,
    )
    del traindata
    gc.collect()
    print("Data processing took {}s".format(time.time() - T1))
    T1 = time.time()

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
    print("Computing residuals took {}s".format(time.time() - T1))
    T1 = time.time()

    x_residuals = {
        sig: xres[
            ..., i * scenario["profile_length"] : (i + 1) * scenario["profile_length"]
        ]
        for i, sig in enumerate(scenario["profile_names"])
    }

    evaluation_metrics = {}
    for metric_name, metric in metrics.items():
        s = 0
        key = "linear_sys_" + metric_name
        val = metric(lres)
        evaluation_metrics[key] = val
        for sig in scenario["profile_names"]:
            key = sig + "_" + metric_name
            val = metric(x_residuals[sig])
            s += val / len(scenario["profile_names"])
            evaluation_metrics[key] = val
        evaluation_metrics["coder_" + metric_name] = s

    print("Computing metrics took {}s".format(time.time() - T1))
    T1 = time.time()

    encoder_data = helpers.lran_helpers.compute_encoder_data(
        model, scenario, valdata, verbose=0
    )
    del valdata
    gc.collect()
    print("Computing encoder data took {}s".format(time.time() - T1))
    T1 = time.time()

    encoder_norm_data = helpers.lran_helpers.compute_norm_data(
        encoder_data["x0"], encoder_data["z0"]
    )
    decoder_norm_data = helpers.lran_helpers.compute_norm_data(
        encoder_data["xp0"], encoder_data["zp0"]
    )

    print("Computing norm data took {}s".format(time.time() - T1))
    T1 = time.time()

    # store operator norm data
    evaluation_metrics["p25_encoder_operator_norm"] = np.nanpercentile(
        encoder_norm_data["operator_norm"], 25
    )
    evaluation_metrics["p50_encoder_operator_norm"] = np.nanpercentile(
        encoder_norm_data["operator_norm"], 50
    )
    evaluation_metrics["p75_encoder_operator_norm"] = np.nanpercentile(
        encoder_norm_data["operator_norm"], 75
    )
    evaluation_metrics["p99_encoder_operator_norm"] = np.nanpercentile(
        encoder_norm_data["operator_norm"], 99
    )

    evaluation_metrics["p25_decoder_operator_norm"] = np.nanpercentile(
        decoder_norm_data["operator_norm"], 25
    )
    evaluation_metrics["p50_decoder_operator_norm"] = np.nanpercentile(
        decoder_norm_data["operator_norm"], 50
    )
    evaluation_metrics["p75_decoder_operator_norm"] = np.nanpercentile(
        decoder_norm_data["operator_norm"], 75
    )
    evaluation_metrics["p99_decoder_operator_norm"] = np.nanpercentile(
        decoder_norm_data["operator_norm"], 99
    )

    evaluation_metrics["p25_encoder_lipschitz_constant"] = np.nanpercentile(
        encoder_norm_data["lipschitz_constant"], 25
    )
    evaluation_metrics["p50_encoder_lipschitz_constant"] = np.nanpercentile(
        encoder_norm_data["lipschitz_constant"], 50
    )
    evaluation_metrics["p75_encoder_lipschitz_constant"] = np.nanpercentile(
        encoder_norm_data["lipschitz_constant"], 75
    )
    evaluation_metrics["p99_encoder_lipschitz_constant"] = np.nanpercentile(
        encoder_norm_data["lipschitz_constant"], 99
    )

    evaluation_metrics["p25_decoder_lipschitz_constant"] = np.nanpercentile(
        decoder_norm_data["lipschitz_constant"], 25
    )
    evaluation_metrics["p50_decoder_lipschitz_constant"] = np.nanpercentile(
        decoder_norm_data["lipschitz_constant"], 50
    )
    evaluation_metrics["p75_decoder_lipschitz_constant"] = np.nanpercentile(
        decoder_norm_data["lipschitz_constant"], 75
    )
    evaluation_metrics["p99_decoder_lipschitz_constant"] = np.nanpercentile(
        decoder_norm_data["lipschitz_constant"], 99
    )

    # time dependent residuals in z, x
    for metric_name, metric in metrics.items():
        evaluation_metrics["dzrel_" + metric_name] = np.array(
            [
                metric(encoder_data["dz"][:, i, :])
                / np.nanmedian(encoder_norm_data["z0_norm"])
                for i in range(encoder_data["dz"].shape[1])
            ]
        )
        evaluation_metrics["dz_" + metric_name] = np.array(
            [
                metric(encoder_data["dz"][:, i, :])
                for i in range(encoder_data["dz"].shape[1])
            ]
        )
        evaluation_metrics["dxrel_" + metric_name] = np.array(
            [
                metric(encoder_data["dx"][:, i, :])
                / np.nanmedian(encoder_norm_data["x0_norm"])
                for i in range(encoder_data["dx"].shape[1])
            ]
        )
        evaluation_metrics["dx_" + metric_name] = np.array(
            [
                metric(encoder_data["dx"][:, i, :])
                for i in range(encoder_data["dx"].shape[1])
            ]
        )

    scenario["encoder_norm_data"] = encoder_norm_data
    scenario["decoder_norm_data"] = decoder_norm_data
    scenario["evaluation_metrics"] = evaluation_metrics

    for key, val in evaluation_metrics.items():
        print(key)
        print(val)

    T1 = time.time()

    with open(file_path, "wb+") as f:
        pickle.dump(copy.deepcopy(scenario), f)
    print("Repickling took {}s".format(time.time() - T1))
    print("TOTAL took {}s".format(time.time() - T0))
    gc.collect()


if __name__ == "__main__":
    args = sys.argv[1:]
    nargs = len(args)
    if nargs < 11:
        for arg in args:
            evaluate(os.path.abspath(arg))
            gc.collect()
    else:
        for i in range(0, nargs, 10):
            job = "eval_" + args[i].split("/")[-1]
            base_path = "/".join(os.path.abspath(args[i]).split("/")[:-1]) + "/"
            file_path = base_path + job + ".slurm"
            paths = [os.path.abspath(arg) for arg in args[i : i + 10]]
            command = "module purge \n"
            command += "module load anaconda \n"
            command += "conda activate tf2-gpu \n"
            command += "\n".join(
                [
                    "python ~/plasma-profile-predictor/evaluate_autoencoder.py " + path
                    for path in paths
                ]
            )
            slurm_script(
                file_path=file_path,
                command=command,
                job_name=job,
                ncpu=1,
                ngpu=0,
                mem=60,
                time=235,
                user="",
            )
        print("Jobs submitted, exiting")
