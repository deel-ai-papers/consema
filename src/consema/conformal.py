import numpy as np
import torch
from typing import Optional, List, Union
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm

from benchmarkerie.models import (
    UniversegInferenceWrap,
    PranetPolypsPrecomputedInferencer,
)
from consema.morphology import (
    operator_thresholding,
    operator_dilation_disk_radius,
    operator_dilation_sequential,
    dilation_score_variable_disk,
    dilation_score_fixed_disk,
    dilation_metrics,
)


def recovered_pixels_bin_array(
    gt: np.ndarray, pred_mask: np.ndarray, extended_mask: np.ndarray
) -> np.ndarray:
    recovered = np.where((gt == 1) & (pred_mask == 0) & (extended_mask == 1), 1, 0)
    return recovered


def thresholding_score(
    gt_mask_: np.ndarray,
    soft_pred_mask_: np.ndarray,
    coverage_threshold: float,
    se_params_=None,
    return_dilated_mask=False,
):
    if se_params_ is not None:
        se_params_ = None

    num_pixels_gt = np.count_nonzero(gt_mask_)

    ## REMARK: if threshold in (0,1), than we could correct "negatively" too (thre < 0.5).
    ## Hence, we could contract or expand the mask, and the conformal quantile could be negative.
    #
    def eval_thre_coverage(thre):
        dilated_mask = operator_thresholding(
            pred_softmax=soft_pred_mask_, operator_parameter=thre
        )

        coverage_array = np.multiply(dilated_mask, gt_mask_)
        coverage = np.sum(coverage_array) / num_pixels_gt
        return coverage

    optimal_thre = dichotomic_search(eval_thre_coverage, coverage_threshold)

    nonconformity_score = 1 - optimal_thre

    if return_dilated_mask:
        dilated_mask = operator_thresholding(
            pred_softmax=soft_pred_mask_, operator_parameter=optimal_thre
        )
        return nonconformity_score, dilated_mask

    return nonconformity_score


def dichotomic_search(
    eval_monotone_decreasing_func,
    objective_lower_bound,
    lbd_lower=0,
    lbd_upper=1,
    n_iter=50,
    max_opt_gap=1e-32,
):
    for _ in range(n_iter):
        if abs(lbd_upper - lbd_lower) < max_opt_gap:
            break

        mid = (lbd_upper + lbd_lower) / 2
        eff = eval_monotone_decreasing_func(mid)

        if eff > objective_lower_bound:
            lbd_lower = mid
        else:
            lbd_upper = mid

    return lbd_lower


from typing import Union


@dataclass
class ConformalResults:
    conformal_tests: List[float]
    empirical_covratios: List[float]
    added_pixels: List[int]
    stretch_factors: List[float]
    conformalizing_quantile: Union[float, int]

    def __init__(
        self,
        conformal_tests=None,
        empirical_covratios=None,
        added_pixels=None,
        stretch_factors=None,
        conformalizing_quantile=None,
    ):
        self.conformal_tests = conformal_tests if conformal_tests is not None else []
        self.empirical_covratios = (
            empirical_covratios if empirical_covratios is not None else []
        )
        self.added_pixels = added_pixels if added_pixels is not None else []
        self.stretch_factors = stretch_factors if stretch_factors is not None else []
        self.conformalizing_quantile = (
            conformalizing_quantile if conformalizing_quantile is not None else 0
        )

        # check that all lists are of the same length. raise error if not
        if not all(
            len(x) == len(self.conformal_tests)
            for x in [
                self.empirical_covratios,
                self.added_pixels,
                self.stretch_factors,
            ]
        ):
            raise ValueError("All lists must be of the same length")

    ## since all lists are of the same length, we can access any element by index
    def __getitem__(self, idx):
        return (
            self.conformal_tests[idx],
            self.empirical_covratios[idx],
            self.added_pixels[idx],
            self.stretch_factors[idx],
        )


class Conformalizer:

    def __init__(
        self,
        inferencer: Union[UniversegInferenceWrap, PranetPolypsPrecomputedInferencer],
        nonconformity_function_name: str,
        structuring_element_params: Optional[dict],
        verbose: bool = False,
    ) -> None:
        self.inferencer = inferencer
        self.verbose = verbose

        if nonconformity_function_name == "fixed_disk":
            self.nonconformity_function = dilation_score_fixed_disk
            self.operator = operator_dilation_sequential
        elif nonconformity_function_name == "variable_disk":
            self.nonconformity_function = dilation_score_variable_disk
            self.operator = operator_dilation_disk_radius
        elif nonconformity_function_name == "thresholding":
            self.nonconformity_function = thresholding_score
            self.operator = operator_thresholding
        else:
            raise ValueError("nonconformity_function_name not recognized")

        self.structuring_element_params = structuring_element_params
        if self.nonconformity_function.__name__ == "thresholding_score":
            self.structuring_element_params = None

        self.alphas = []

    def compute_nonconformity_scores(
        self,
        calibration_dataset,
        coverage_ratio,
    ):
        self.results = defaultdict(list)
        nonconformity_scores_list = []

        for i, calib_elem in tqdm(
            enumerate(calibration_dataset), disable=not self.verbose
        ):
            try:
                image, label = calib_elem
                precomputed_soft_prediction = None
            except:
                try:
                    image, label, precomputed_soft_prediction = calib_elem
                except:
                    raise ValueError(
                        "Calibration data must be a tuple (2 or 3 elements, if precomputed preds)"
                    )

            if isinstance(self.inferencer, UniversegInferenceWrap):
                precomputed_soft_prediction = None
                vals = self.inferencer.inference(
                    image, label, precomputed_soft_prediction
                )
            else:
                vals = self.inferencer.inference(
                    image, label, precomputed_soft_prediction
                )

            for k, v in vals.items():
                self.results[k].append(v)

            gt_mask = vals["Ground Truth"][0]
            pred_mask = vals["Prediction"][0]

            if self.nonconformity_function.__name__ == "thresholding_score":
                pred_mask = vals["Soft Prediction"][0]

            nc_score = self.nonconformity_function(
                gt_mask,
                pred_mask,
                se_params_=self.structuring_element_params,
                coverage_threshold=coverage_ratio,
            )
            self.results["nonconformity"].append(nc_score)
            nonconformity_scores_list.append(nc_score)

        if self.inferencer.return_numpy:
            self.nonconformity_scores = np.array(nonconformity_scores_list)
        else:
            self.nonconformity_scores = torch.Tensor(nonconformity_scores_list)

    def test_inferences(self, test_dataset, verbose: bool = False):
        results = []

        for test_datum in tqdm(test_dataset, disable=not verbose):
            try:
                image, label = test_datum
                precomputed_soft_prediction = None
            except:
                try:
                    image, label, precomputed_soft_prediction = test_datum
                except:
                    raise ValueError(
                        "Test data must be a tuple (2 or 3 elements, if precomputed preds)"
                    )

            results.append(
                self.inferencer.inference(image, label, precomputed_soft_prediction)
            )

        return results

    def compute_conforming_idx(self, alpha):
        if self.nonconformity_scores is None:
            raise ValueError("No nonconformity scores available")

        nc_scores = self.nonconformity_scores

        try:
            conformal_idx = np.ceil((1.0 - alpha) * (len(nc_scores) + 1)).astype(int)
        except ValueError:
            print(
                f"{alpha = } too small for {len(nc_scores) = }. alpha must be > {1/len(nc_scores)}"
            )

        return conformal_idx

    def get_conformal_quantile(self, alpha):
        conformal_idx = self.compute_conforming_idx(alpha)
        conformal_quantile = sorted(self.nonconformity_scores.tolist())[conformal_idx]
        return conformal_quantile

    def test_conformalization(
        self, test_inference_results, alpha_risk: float, covratio: float
    ):
        conformalizing_quantile = self.get_conformal_quantile(alpha_risk)

        results = ConformalResults([], [], [], [], [])
        results.conformalizing_quantile = conformalizing_quantile

        for _, pred in enumerate(test_inference_results):
            if isinstance(pred["Ground Truth"][0], torch.Tensor):
                test_gt_mask = pred["Ground Truth"][0].cpu().numpy()
                test_hard_pred = pred["Prediction"][0].cpu().numpy()
                test_soft_pred = pred["Soft Prediction"][0].cpu().numpy()
            else:
                test_gt_mask = pred["Ground Truth"][0]
                test_hard_pred = pred["Prediction"][0]
                test_soft_pred = pred["Soft Prediction"][0]

            if self.nonconformity_function.__name__ == "thresholding_score":
                dilated_mask = self.operator(
                    test_soft_pred,
                    1 - conformalizing_quantile,
                )
            else:
                dilated_mask = self.operator(
                    test_hard_pred,
                    conformalizing_quantile,
                    self.structuring_element_params,
                )

            empirical_covratio = np.sum(dilated_mask * test_gt_mask) / np.sum(
                test_gt_mask
            )
            results.empirical_covratios.append(empirical_covratio)

            binary_test = empirical_covratio >= covratio
            results.conformal_tests.append(binary_test)

            metrics = dilation_metrics(dilated_mask, test_hard_pred)
            results.added_pixels.append(metrics[0])
            results.stretch_factors.append(metrics[1])

        return results

    def plot_nc_scores(self, alpha_risk, threshold_score: bool = False):
        if self.nonconformity_scores is None:
            raise ValueError(
                "Nonconformity scores unavailable: run [self.compute_nonconformity_scores(...)]"
            )
        conformal_idx = self.compute_conforming_idx(alpha_risk)
        conformal_q = self.get_conformal_quantile(alpha_risk)

        plt.figure(figsize=(5, 3))
        plt.plot(sorted(self.nonconformity_scores.tolist()), label="nc scores")
        plt.scatter(conformal_idx, conformal_q, color="red")

        if threshold_score:
            threshos = (self.nonconformity_scores).tolist()
            plt.plot(sorted(threshos, reverse=True), label="inference thresholds")
            plt.scatter(conformal_idx, conformal_q, color="red")
            plt.axhline(y=0.5, color="red", label="default threshold: 0.5")
            plt.axhline(
                y=conformal_q,
                color="green",
                label=f"cp threshold: f(x) > {conformal_q}",
            )

        plt.ylabel("Value of nonconformity score")
        plt.xlabel("indices of sorted nc scores")
        plt.legend()
        plt.show()

    def plot_nc_scores_frequency(self, alpha: Optional[float] = None):
        plt.figure(figsize=(5, 3))
        if alpha is not None:
            conformal_idx = self.compute_conforming_idx(alpha)
            conformalizing_quantile = sorted(self.nonconformity_scores.tolist())[
                conformal_idx
            ]
            plt.axvline(x=conformalizing_quantile, color="red", label="alpha")

        if self.nonconformity_function.__name__ == "thresholding_score":
            plt.hist(self.nonconformity_scores, bins=20)
            plt.xlabel("Nonconformity Scores (1 - threshold)")
            plt.ylabel("Frequency")
        else:
            score_counts = {}  # Count the frequency of each score
            for score in self.nonconformity_scores:
                if score in score_counts:
                    score_counts[score] += 1
                else:
                    score_counts[score] = 1

            sorted_scores = sorted(score_counts.keys())
            plt.bar(sorted_scores, [score_counts[score] for score in sorted_scores])
            plt.xlabel("Nonconformity Score")
            plt.ylabel("Frequency")
            plt.title("Frequency of Nonconformity Scores")
            plt.show()

        plt.show()
