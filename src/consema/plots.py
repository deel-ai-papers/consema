import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import matplotlib.patches as mpatches
from typing import Literal, Optional
import skimage.morphology as morpho


class PredictionPlot:
    def __init__(self, image, gt_mask, pred_mask, pred, figsize=(12, 2)):
        # Ensure all inputs are either numpy arrays or torch tensors
        # Convert all torch tensors to numpy arrays
        self.image = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        self.gt_mask = (
            gt_mask.cpu().numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask
        )
        self.pred_mask = (
            pred_mask.cpu().numpy()
            if isinstance(pred_mask, torch.Tensor)
            else pred_mask
        )
        self.pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred

        # Ensure all inputs have the same size
        if not (
            self.image.shape
            == self.gt_mask.shape
            == self.pred_mask.shape
            == self.pred.shape
        ):
            print(f"{self.image.shape = }")
            print(f"{self.gt_mask.shape = }")
            print(f"{self.pred_mask.shape = }")
            print(f"{self.pred.shape = }")
            raise ValueError("All inputs must have the same size")

        self.fig, self.axs = plt.subplots(1, 4, figsize=figsize)
        self._create_plot()

    def _create_plot(self):
        # Build the plots
        self.axs[0].set_title("Input image")
        self.axs[0].imshow(self.image, cmap="grey", interpolation="none")
        #
        self.axs[1].set_title("gt mask")
        self.axs[1].imshow(self.gt_mask, interpolation="none")
        #
        self.axs[2].set_title("pred mask")
        self.axs[2].imshow(self.pred_mask, interpolation="none")
        #
        self.axs[3].set_title("Soft prediction")
        im = self.axs[3].imshow(self.pred, cmap="grey", interpolation="none")
        self.fig.colorbar(im, ax=self.axs[3])

    def show(self):
        self.fig.show()

    def save(self, path):
        self.fig.savefig(path)


def visualize_false_negatives(
    input_image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
):
    # # if input_image.ndim != 2 or gt_mask.ndim != 2 or pred_mask.ndim != 2:
    # #     raise ValueError("All inputs must be 2D arrays of shape [H x W]")
    # if input_image.ndim not in [2, 3] or gt_mask.ndim != 2 or pred_mask.ndim != 2:
    #     raise ValueError(
    #         "gt_mask and pred_mask must be 2D arrays of shape [H x W], and input_image must be either 2D or 3D with an RGB channel"
    #     )
    # if input_image.ndim == 3 and input_image.shape[2] not in [1, 3]:
    #     raise ValueError(
    #         "If input_image is 3D, it must have 3 channels (RGB or grayscale)"
    #     )

    bintruth = gt_mask.copy().astype(int)
    binpred = pred_mask.copy().astype(int)
    output = binpred.copy()

    truepos_val = 1
    truepos = (bintruth * binpred) * truepos_val
    falseneg_val = 2
    falseneg = (bintruth * (1 - binpred)) * falseneg_val
    falsepos_val = 3
    falsepos = ((1 - bintruth) * binpred) * falsepos_val

    where_truepos = np.where(truepos > 0)
    where_falseneg = np.where(falseneg > 0)
    where_falsepos = np.where(falsepos > 0)

    output[where_truepos] = 1
    output[where_falseneg] = 2
    output[where_falsepos] = 3

    c = sns.color_palette()
    bg_color = np.array([234, 226, 183]) / 255  # light yellow
    palette = np.array(
        [
            bg_color,
            c[2],
            c[3],
            c[0],
        ]
    )

    rgb = palette[output]

    params = {
        "cmap": "Greys",
        "interpolation": "none",
    }

    labels = {0: "TN", 1: "TP (pred)", 2: "FN", 3: "FP (pred)"}

    ncols = 3 if input_image is not None else 2
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(2.5 * ncols, 3))
    axes[0].imshow(palette[bintruth], **params)
    axes[0].set_title("Ground-truth")
    axes[1].imshow(rgb, **params)
    axes[1].set_title("Preds (green & blue)")
    if input_image is not None:
        axes[2].imshow(input_image, **params)
        axes[2].set_title("Input Image")

    patches = [
        mpatches.Patch(color=palette[i], label=labels[i])
        for i, val in enumerate(np.unique(output))
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()


def plot_margin_and_recovered(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    dilated_mask: np.ndarray,
    margin: Optional[np.ndarray] = None,
    input_image: Optional[np.ndarray] = None,
    softprediction: Optional[np.ndarray] = None,
    plot_hard_margin: Optional[bool] = False,
    figsize: Optional[tuple] = None,
):
    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.cpu().numpy()
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.cpu().numpy()
    if isinstance(dilated_mask, torch.Tensor):
        dilated_mask = dilated_mask.cpu().numpy()

    if margin is None:
        margin = np.logical_xor(dilated_mask, predicted_mask)
    recovered = np.where(
        (ground_truth_mask == 1) & (predicted_mask == 0) & (dilated_mask == 1), 1, 0
    )
    not_recovered = np.where(
        (ground_truth_mask == 1) & (predicted_mask == 0) & (dilated_mask == 0), 666, 0
    )

    # TODO: IMPLEMENT DILATION OF BACKGROUND
    if recovered.sum() == 0:
        print("No pixels were recovered")
        # return

    recovered_rgba = np.zeros((*recovered.shape, 4))
    # (X,X,X,1) for 1 with full opacity
    recovered_rgba[recovered == 1] = [1, 0, 0, 1]
    recovered_rgba[recovered == 0] = [0, 0, 0, 0]  # Transparent for value 0
    recovered_rgba[not_recovered == 666] = [0, 100 / 255, 0, 1]

    if plot_hard_margin:
        margin = np.where(margin > 0, 1, 0)

    nplots = 3
    if input_image is not None:
        nplots += 1
    if softprediction is not None:
        nplots += 1

    figsize = (2.5 * nplots, 3) if figsize is None else figsize
    _, axes = plt.subplots(1, nplots, figsize=figsize)
    axes[0].imshow(ground_truth_mask, cmap="gray", interpolation="none")
    axes[0].set_title("Ground-truth mask")

    axes[1].imshow(predicted_mask, cmap="gray", interpolation="none")
    axes[1].set_title("Pred mask")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(margin, cmap="Blues", interpolation="none")
    axes[2].imshow(recovered_rgba)
    axes[2].set_title("Margin + Recovered")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    if softprediction is not None and input_image is None:
        axes[3].imshow(softprediction, cmap="gray", interpolation="none")
        axes[3].set_title("Prediction scores")
        axes[3].set_xticks([])
        axes[3].set_yticks([])
    elif softprediction is not None and input_image is not None:
        axes[3].imshow(softprediction, cmap="gray", interpolation="none")
        axes[3].set_title("Prediction scores")
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        plt.subplots_adjust(left=0.2)
        axes[4].imshow(input_image, cmap="gray", interpolation="none")
        axes[4].set_title("Input Image")
    elif input_image is not None and softprediction is None:
        axes[3].imshow(input_image, cmap="gray", interpolation="none")
        axes[3].set_title("Input Image")

    plt.show()


def margin_gradient_visu(
    input_mask: np.ndarray,
    pred_set_type: Literal["fixed_disk", "variable_disk"],
    operator_parameter,
    offset: Optional[int] = None,
    plot_darker_input_mask: Optional[bool] = False,
    se_params_: Optional[dict] = None,
) -> np.ndarray:
    """computes the margin via dilation with fixed-size or variable-size disk

    The margin returns should be of the same shape as the that obtained
    from the computation of nonconformity score, except that here the
    margin has values 1 to [operator_parameter] (inclusive), so that
    one can visualize the iterative accretion of the margin, that is,
    how it grows while the disk is enlarged.
    """
    if offset is None:
        offset = int(operator_parameter / 2)

    if se_params_ is None:
        raise ValueError("se_params_ must be provided for morpho dilation")

    radius = 1
    dilated = input_mask.copy()
    gradient_margin = np.zeros_like(input_mask, dtype=int)

    if pred_set_type == "variable_disk":
        for it_ in range(1, operator_parameter + 1):
            structuring_element = morpho.disk(it_, **se_params_)
            old_dilated = dilated.copy()
            dilated = morpho.binary_dilation(input_mask, structuring_element).astype(
                int
            )
            margin = (dilated - old_dilated) * (operator_parameter + 1 - it_)
            gradient_margin = gradient_margin + margin
    elif pred_set_type == "fixed_disk":
        dilated = input_mask.copy()
        for it_ in range(operator_parameter):
            structuring_element = morpho.disk(radius, **se_params_)
            old_dilated = dilated.copy()
            dilated = morpho.binary_dilation(dilated, structuring_element).astype(int)
            margin = (dilated - old_dilated) * (operator_parameter + 1 - it_)
            gradient_margin = gradient_margin + margin

    # make the margin stand out against the background (for better visualization)
    gradient_margin[gradient_margin > 0] += offset  # int(operator_parameter / 2)

    ## make input mask stand out
    if plot_darker_input_mask:
        gradient_margin[input_mask > 0] = 1.25 * np.max(gradient_margin)

    return gradient_margin
