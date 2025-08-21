import os
import numpy as np
import torch
import csv
import argparse

from typing import Literal
from dotenv import load_dotenv


from consema.conformal import Conformalizer
from benchmarkerie.datasets import (
    setup_polyps,
    make_universeg_predictions,
    setup_wbc,
    setup_oasis,
)
from benchmarkerie.models import (
    PranetPolypsPrecomputedInferencer,
    UniversegInferenceWrap,
)
from collections import namedtuple

from universeg import universeg  # installed via Makefile

ExperimentData = namedtuple(
    "ExperimentData",
    [
        "data_calib",
        "data_test",
        "inferencer",
        "chosen_nc_score",
        "se_params",
        "alpha",
        "covratio",
    ],
)

ExperimentResults = namedtuple(
    "ExperimentResults",
    ["empirical_coverage", "empirical_stretch"],
)


def setup_pranet_polyps(config: dict):
    assert (
        config["dataset"] == "polyps" and config["model"] == "pranet"
    ), "Invalid config: expected polyps and pranet"

    device_str = config["device"] if torch.cuda.is_available() else "cpu"
    inferencer = PranetPolypsPrecomputedInferencer(
        device=device_str
    )  # device_str is not used by PranetPolypsPrecomputedInferencer
    (
        calib_images,
        calib_gt_arrays,
        calib_pred_arrays,
        test_images,
        test_gt_arrays,
        test_pred_arrays,
    ) = setup_polyps(config["random_seed"], 0.5)

    data_calib = tuple(
        (img, gt, softpred)
        for img, gt, softpred in zip(calib_images, calib_gt_arrays, calib_pred_arrays)
    )
    data_test = tuple(
        (img, gt, softpred)
        for img, gt, softpred in zip(test_images, test_gt_arrays, test_pred_arrays)
    )
    # print(f" --- n calibration points: {len(data_calib)}")

    return ExperimentData(
        data_calib,
        data_test,
        inferencer,
        config["chosen_nc_score"],
        config["se_params"],
        config["alpha"],
        config["covratio"],
    )


def setup_universeg_wbc(config: dict):
    ## FIXME: put automagic stuff here
    PREDS_DIR = None

    assert (
        config["dataset"] == "wbc" and config["model"] == "universeg"
    ), "Invalid config: expected wbc and universeg"

    device_str = config["device"] if torch.cuda.is_available() else "cpu"

    model = universeg(pretrained=True)
    model.to(device_str)

    LABEL = "nucleus"
    # LABEL = "cytoplasm" ## <-- not particularly interesting

    if (
        config["label"] is not None
    ):  # later addition: not used for expes in paper (used LABEL = "nucleus")
        LABEL = config["label"]
        print(f" --- Using label {LABEL} for WBC dataset")

    ## high performance predictor
    # n_support_samples = 48
    ## medium performance predictor
    n_support_samples = 24
    # ## lower performance predictor
    # n_support_samples = 12

    _, input_data_calib, input_data_test, support_images, support_labels = setup_wbc(
        LABEL,
        n_support_samples,
        random_seed=config["random_seed"],
        device=device_str,
    )

    inferencer = UniversegInferenceWrap(
        model=model,
        support_images=support_images,
        support_labels=support_labels,
        device=device_str,
    )

    calib_images, calib_gt_arrays, calib_pred_arrays = make_universeg_predictions(
        input_data_calib,
        f"calib_nsup_{n_support_samples}",
        device_str=device_str,
        support_images=support_images,
        support_labels=support_labels,
        universeg_model=model,
        save_to=PREDS_DIR,
    )

    test_images, test_gt_arrays, test_pred_arrays = make_universeg_predictions(
        input_data_test,
        f"test_nsup_{n_support_samples}",
        device_str=device_str,
        support_images=support_images,
        support_labels=support_labels,
        universeg_model=model,
        save_to=PREDS_DIR,
    )

    data_calib = tuple(
        (img, gt, softpred)
        for img, gt, softpred in zip(calib_images, calib_gt_arrays, calib_pred_arrays)
    )
    data_test = tuple(
        (img, gt, softpred)
        for img, gt, softpred in zip(test_images, test_gt_arrays, test_pred_arrays)
    )

    return ExperimentData(
        data_calib,
        data_test,
        inferencer,
        config["chosen_nc_score"],
        config["se_params"],
        config["alpha"],
        config["covratio"],
    )


def setup_universeg_oasis(config: dict):
    PREDS_DIR = None

    assert (
        config["dataset"] == "oasis" and config["model"] == "universeg"
    ), "Invalid config: expected oasis and universeg"

    device_str = config["device"] if torch.cuda.is_available() else "cpu"

    model = universeg(pretrained=True)
    model.to(device_str)

    # all other labels are also available, and give similar results
    LABEL = (
        0  # background: not the most interesting label, but the one used in the paper
    )

    if (
        config["label"] is not None
    ):  # later addition: was not used for expes in paper (used LABEL = 0)
        LABEL = config["label"]
        print(f" --- Using label {LABEL} for OASIS dataset")

    ## high performance predictor
    # n_support_samples = 48
    ## medium performance predictor
    n_support_samples = 24
    # ## lower performance predictor
    # n_support_samples = 12

    _, input_data_calib, input_data_test, support_images, support_labels = setup_oasis(
        label=LABEL,
        n_support_samples=n_support_samples,
        random_seed=config["random_seed"],
        device=device_str,
    )

    inferencer = UniversegInferenceWrap(
        model=model,
        support_images=support_images,
        support_labels=support_labels,
        device=device_str,
    )

    calib_images, calib_gt_arrays, calib_pred_arrays = make_universeg_predictions(
        input_data_calib,
        f"calib_nsup_{n_support_samples}",
        device_str=device_str,
        support_images=support_images,
        support_labels=support_labels,
        universeg_model=model,
        save_to=PREDS_DIR,
    )

    test_images, test_gt_arrays, test_pred_arrays = make_universeg_predictions(
        input_data_test,
        f"test_nsup_{n_support_samples}",
        device_str=device_str,
        support_images=support_images,
        support_labels=support_labels,
        universeg_model=model,
        save_to=PREDS_DIR,
    )

    data_calib = tuple(
        (img, gt, softpred)
        for img, gt, softpred in zip(calib_images, calib_gt_arrays, calib_pred_arrays)
    )
    data_test = tuple(
        (img, gt, softpred)
        for img, gt, softpred in zip(test_images, test_gt_arrays, test_pred_arrays)
    )

    return ExperimentData(
        data_calib,
        data_test,
        inferencer,
        config["chosen_nc_score"],
        config["se_params"],
        config["alpha"],
        config["covratio"],
    )


def run_experiment(experiment_data: ExperimentData):
    data_calib = experiment_data.data_calib
    data_test = experiment_data.data_test
    inferencer = experiment_data.inferencer
    chosen_nc_score = experiment_data.chosen_nc_score
    se_params = experiment_data.se_params
    alpha = experiment_data.alpha
    covratio = experiment_data.covratio

    cpred = Conformalizer(
        inferencer=inferencer,
        nonconformity_function_name=chosen_nc_score,
        structuring_element_params=se_params,
    )
    covratio = experiment_data.covratio
    cpred.compute_nonconformity_scores(data_calib, covratio)

    del data_calib

    _test_preds = cpred.test_inferences(data_test)
    test_results = cpred.test_conformalization(_test_preds, alpha, covratio)

    del _test_preds

    empirical_coverage = np.mean(test_results.conformal_tests)
    empirical_avg_stretch = np.mean(test_results.stretch_factors)

    results_one_run = (
        empirical_coverage,
        empirical_avg_stretch,
        test_results.conformalizing_quantile,
    )

    return results_one_run


def setup_experiment(config: dict):
    if config["dataset"] == "polyps" and config["model"] == "pranet":
        return setup_pranet_polyps(config)
    elif config["dataset"] == "wbc" and config["model"] == "universeg":
        return setup_universeg_wbc(config)
    elif config["dataset"] == "oasis" and config["model"] == "universeg":
        return setup_universeg_oasis(config)
    else:
        raise ValueError("Unknown dataset or model")


def load_configurations(csv_path):
    configs = []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # row is already a dict with keys matching the CSV headers
            # Convert numeric fields if needed
            row["covratio"] = float(row["covratio"])
            row["alpha"] = float(row["alpha"])

            if row["structuring_element"] == "cross":
                row["se_params"] = dict(strict_radius=True)  # [3 X 3] cross
            elif row["structuring_element"] == "threshold":
                # ignored for thresholding:
                row["se_params"] = None  # dict(strict_radius=True)
            else:
                raise ValueError("ERROR: structuring element not implemented")
            configs.append(row)
    return configs


def run_seed(
    rnd_seed: int,
    dataset: Literal["polyps", "wbc", "oasis", "threshold"],
    timestamp: str,
):
    load_dotenv()
    configs_dir = os.getenv("CONFIGS_EXPERIMENTS")

    is_thresholdo = False

    if dataset == "threshold":
        is_thresholdo = True
        config_file_path = f"{configs_dir}/thresholding/config_polyps_threshold.csv"
        dataset = "polyps"
        print(f"--- Run THRESHOLDING experiment on polyps dataset")
    else:
        config_file_path = f"{configs_dir}/config_{dataset}.csv"

    configs = load_configurations(config_file_path)

    print(f"--- Experiment: dataset = {dataset}, random seed = {rnd_seed}")

    results = []

    successful_configs = []

    for cfg in configs:
        try:
            cfg["random_seed"] = rnd_seed
            experiment_data = setup_experiment(cfg)
            coverage, stretch, cp_quantile = run_experiment(experiment_data)

            # Export all fields in the config file in addition to what already done
            result_row = {
                **cfg,
                "coverage": coverage,
                "stretch": stretch,
                "cp_quantile": cp_quantile,
            }
            results.append(result_row)
            successful_configs.append(cfg)
        except Exception as e:
            print(f" --- Error: {e}")
            traceback.print_exc()

    # Write successful configs to a file
    if is_thresholdo:
        print(f" +++ {is_thresholdo=}")
        successful_configs_file = f"{os.getenv('RESULTS_DIR')}/successful_configs_config_{dataset}_threshold_{timestamp}.csv"
    else:
        successful_configs_file = f"{os.getenv('RESULTS_DIR')}/successful_configs_config_{dataset}_{timestamp}.csv"

    file_exists = os.path.isfile(successful_configs_file)
    with open(
        successful_configs_file,
        mode="a" if file_exists else "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=successful_configs[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(successful_configs)

    return results


import traceback
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--dataset", type=str, choices=["polyps", "wbc", "oasis"], required=True
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Use threshold on sigmoid of polyps-pranet",
    )
    args = parser.parse_args()

    dataset = args.dataset
    if args.threshold:
        dataset = "threshold"

    semi_randomici = [i for i in range(1989, 2025)]

    outputs = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        outputs = []
        for seed in semi_randomici:
            outputs.extend(run_seed(seed, dataset, timestamp))

        results_dir = os.getenv("RESULTS_DIR")

        output_file = f"{results_dir}/results_{dataset}_fixed_disk_{timestamp}.csv"
        if args.threshold:
            output_file = f"{results_dir}/results_{dataset}_threshold_{timestamp}.csv"

        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=outputs[0].keys())
            writer.writeheader()
            writer.writerows(outputs)

    except Exception as e:
        print(f" --- Error: {e}")
        traceback.print_exc()  # prints the full traceback


if __name__ == "__main__":
    main()
