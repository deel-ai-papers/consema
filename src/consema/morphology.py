import numpy as np
import torch
import skimage.morphology as morpho

from typing import Optional


def operator_dilation_sequential(
    input_mask: np.ndarray,
    operator_parameter,
    se_params_: Optional[dict] = None,
) -> np.ndarray:
    if se_params_ is None:
        raise ValueError("se_params_ must be provided for morpho dilation")
    radius = 1
    d_se = morpho.disk(radius, **se_params_)  # SE: structuring element

    dilated = input_mask.copy()
    for _ in range(operator_parameter):
        dilated = morpho.binary_dilation(dilated, d_se)

    return dilated


def operator_dilation_disk_radius(
    input_mask: np.ndarray,
    operator_parameter,
    se_params_: Optional[dict] = None,
) -> np.ndarray:
    if se_params_ is None:
        raise ValueError("se_params_ must be provided for morpho dilation")
    radius = operator_parameter
    d_se = morpho.disk(radius, **se_params_)  # SE: structuring element
    dilated = morpho.binary_dilation(input_mask, d_se)

    return dilated


def operator_thresholding(
    pred_softmax: np.ndarray,
    operator_parameter,
    se_params_: Optional[dict] = None,
):
    threshold = operator_parameter
    dilated_mask = pred_softmax > threshold
    return dilated_mask


def dilation_score_fixed_disk(
    gt_mask_: np.ndarray,
    pred_mask_: np.ndarray,
    se_params_: dict,
    coverage_threshold: float,
) -> int:
    if isinstance(gt_mask_, torch.Tensor):
        gt_mask_ = gt_mask_.cpu().numpy()
    if isinstance(pred_mask_, torch.Tensor):
        pred_mask_ = pred_mask_.cpu().numpy()

    if gt_mask_.ndim > 2:
        raise ValueError(
            f" -- [gt_mask_] must have 2 dimensions (H x W), but got {gt_mask_.ndim}"
        )
    if pred_mask_.ndim > 2:
        raise ValueError(
            f" -- [pred_mask_] must have 2 dimensions (H x W), but got {pred_mask_.ndim} dimensions"
        )

    num_pixels_gt = np.count_nonzero(gt_mask_)
    dilated_mask = pred_mask_

    if np.sum(dilated_mask) <= 0:
        raise ValueError(" -- [pred_mask_] must have at least one pixel")
    if np.sum(gt_mask_) <= 0:
        raise ValueError(" -- [gt_mask_] must have at least one pixel")

    # structuring element is FIXED throughout the iterations
    iteration = 0
    nonconformity_score = iteration
    max_score = 2 * np.max(dilated_mask.shape)

    while True:
        if iteration >= max_score:
            dilated_mask = np.ones_like(dilated_mask)
            nonconformity_score = iteration
            print(f"Warning: max iters reached, setting nonconformity_score to 100")
            break

        coverage_tensor = np.multiply(dilated_mask, gt_mask_)
        coverage = np.sum(coverage_tensor) / num_pixels_gt

        if coverage >= coverage_threshold:
            break
        else:
            iteration += 1  # counts iterations, does not affect the radius of disk B
            radius = 1
            # WARNING: we do ONE dilation of SE with radius 1 (3x3 cross or disk)
            dilated_mask = operator_dilation_disk_radius(
                input_mask=dilated_mask,
                operator_parameter=radius,
                se_params_=se_params_,
            )

        nonconformity_score = iteration

    return nonconformity_score


def dilation_score_variable_disk(
    gt_mask_: np.ndarray,
    pred_mask_: np.ndarray,
    se_params_: dict,
    coverage_threshold: float,
) -> int:
    """
    WARNING: this function was not used for experiments in paper, hence
    it was not reviewed and could contain bugs
    """
    if gt_mask_.ndim > 2:
        raise ValueError(
            f" -- [gt_mask_] must have 2 dimensions (H x W), but got {gt_mask_.ndim}"
        )
    if pred_mask_.ndim > 2:
        raise ValueError(
            f" -- [pred_mask_] must have 2 dimensions (H x W), but got {pred_mask_.ndim} dimensions"
        )

    num_pixels_gt = np.count_nonzero(gt_mask_)

    radius = 0
    og_pred_mask = pred_mask_.copy()
    dilated_mask = pred_mask_ 
    nonconformity_score = radius
    max_iter = 35

    while True:
        if radius >= max_iter:
            dilated_mask = np.ones_like(dilated_mask)
            nonconformity_score = 100
            print(f"Warning: max iters reached, setting nonconformity_score to 100")
            break
        coverage_tensor = np.multiply(dilated_mask, gt_mask_)
        coverage = np.sum(coverage_tensor) / num_pixels_gt

        if coverage >= coverage_threshold:
            break
        else:
            radius += 1
            dilated_mask = operator_dilation_disk_radius(
                og_pred_mask, radius, se_params_
            )
            nonconformity_score = radius

    return nonconformity_score


# function to measure the size of the dilated mask and the stretch factor over the original predicted mask
def dilation_metrics(
    dilated_mask: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None,
):
    if dilated_mask.ndim > 2:
        raise ValueError(
            f" -- [dilated_mask] must have 2 dimensions (H x W), but got {dilated_mask.ndim}"
        )
    if predicted_mask.ndim > 2:
        raise ValueError(
            f" -- [predicted_mask] must have 2 dimensions (H x W), but got {predicted_mask.ndim} dimensions"
        )

    num_added_pixels = np.sum(dilated_mask) - np.sum(predicted_mask)
    size_dil_mask = np.sum(dilated_mask)
    stretch = size_dil_mask / np.sum(predicted_mask)
    return num_added_pixels, stretch
