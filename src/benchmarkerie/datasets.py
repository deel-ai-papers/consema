# from dotenv import load_dotenv
from dataclasses import dataclass
import numpy as np
import os, sys
import pathlib
import subprocess
from einops import rearrange
from tqdm import tqdm
import torch
import itertools
from typing import Literal, Optional
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from example_data.wbc import WBCDataset
from example_data.oasis import OASISDataset

REPO_URL_WBC = "https://github.com/zxaoyou/segmentation_WBC.git"


# Load the raw data once and reuse it
def load_polyps_data():
    load_dotenv()
    POLYPS_NPZ = os.getenv("POLYPS_NPZ")
    data = np.load(f"{POLYPS_NPZ}/polyps-pranet.npz")
    sgmd = data["sgmd"]  # sigmoid scores
    gt_masks = data["targets"]
    test_calib_idxs = [elem.item() for elem in data["example_indexes"]]

    # Rescale sigmoid scores to [0, 1] for each image
    sgmd_rescaled = np.array(
        [(img - img.min()) / (img.max() - img.min()) for img in sgmd]
    )
    sgmd = sgmd_rescaled

    return sgmd, gt_masks, test_calib_idxs, POLYPS_NPZ


# DANGER ZONE: this function is called only once to load the data, hence it is a global variable
# Call this function once to load the data
sgmd, gt_masks, test_calib_idxs, POLYPS_NPZ = load_polyps_data()


def setup_polyps(random_seed: int, cal_test_ratio: float):
    np.random.seed(random_seed)
    np.random.shuffle(test_calib_idxs)

    split_num = int(len(test_calib_idxs) * cal_test_ratio)
    calib_idxs = test_calib_idxs[:split_num]
    test_idxs = test_calib_idxs[split_num:]

    assert set(calib_idxs).isdisjoint(
        set(test_idxs)
    ), "calib and test sets are not disjoint"

    from skimage.transform import resize

    calib_pred_arrays = sgmd[calib_idxs]
    calib_gt_arrays = gt_masks[calib_idxs]
    calib_input_img_paths = [
        f"{POLYPS_NPZ}/examples/" + str(rnd_idx_) + ".jpg"
        for rnd_idx_ in test_calib_idxs
    ]
    calib_images = [plt.imread(img_path_) for img_path_ in calib_input_img_paths]

    calib_preds_resized = [
        resize(pred, (img.shape[0], img.shape[1]), anti_aliasing=False)
        for pred, img in zip(calib_pred_arrays, calib_images)
    ]
    calib_gt_resized = [
        resize(gt, (img.shape[0], img.shape[1]), anti_aliasing=False)
        for gt, img in zip(calib_gt_arrays, calib_images)
    ]

    test_pred_arrays = sgmd[test_idxs]
    test_gt_arrays = gt_masks[test_idxs]
    test_input_img_paths = [
        f"{POLYPS_NPZ}/examples/" + str(rnd_idx_) + ".jpg" for rnd_idx_ in test_idxs
    ]
    test_images = [plt.imread(img_path_) for img_path_ in test_input_img_paths]

    test_preds_resized = [
        resize(pred, (img.shape[0], img.shape[1]), anti_aliasing=False)
        for pred, img in zip(test_pred_arrays, test_images)
    ]
    test_gt_resized = [
        resize(gt, (img.shape[0], img.shape[1]), anti_aliasing=False)
        for gt, img in zip(test_gt_arrays, test_images)
    ]

    # add a dummy dimension for batch size (== 1)
    calib_pred_arrays = [np.expand_dims(pred, 0) for pred in calib_pred_arrays]
    calib_gt_arrays = [np.expand_dims(gt, 0) for gt in calib_gt_arrays]
    calib_preds_resized = [np.expand_dims(pred, 0) for pred in calib_preds_resized]
    calib_gt_resized = [np.expand_dims(gt, 0) for gt in calib_gt_resized]
    calib_images = [np.expand_dims(img, 0) for img in calib_images]
    test_pred_arrays = [np.expand_dims(pred, 0) for pred in test_pred_arrays]
    test_gt_arrays = [np.expand_dims(gt, 0) for gt in test_gt_arrays]
    test_preds_resized = [np.expand_dims(pred, 0) for pred in test_preds_resized]
    test_gt_resized = [np.expand_dims(gt, 0) for gt in test_gt_resized]
    test_images = [np.expand_dims(img, 0) for img in test_images]

    return (
        calib_images,
        calib_gt_arrays,
        calib_pred_arrays,
        test_images,
        test_gt_arrays,
        test_pred_arrays,
    )


def check_dataset_splits(data_support, data_calib, data_test):
    """
    Validate that the dataset splits (support, calibration, test) are consistent across datasets.

    Args:
        data_support: The support dataset.
        data_calib: The calibration dataset.
        data_test: The test dataset.

    Raises:
        AssertionError: If any of the splits are not consistent across datasets.
    """
    assert np.array_equal(
        data_support.support_indices, data_calib.support_indices
    ) and np.array_equal(
        data_support.support_indices, data_test.support_indices
    ), "Support indices are not the same across datasets"
    assert np.array_equal(
        data_support.calibration_indices, data_calib.calibration_indices
    ) and np.array_equal(
        data_support.calibration_indices, data_test.calibration_indices
    ), "Calibration indices are not the same across datasets"
    assert np.array_equal(
        data_support.test_indices, data_calib.test_indices
    ) and np.array_equal(
        data_support.test_indices, data_test.test_indices
    ), "Test indices are not the same across datasets"


def setup_wbc(label, n_support_samples, random_seed, device="cpu"):
    data_support = ExtendedWBCDataset(
        dataset="JTSC", split="support", label=label, random_seed=random_seed
    )
    data_calib = ExtendedWBCDataset(
        dataset="JTSC", split="calibration", label=label, random_seed=random_seed
    )
    data_test = ExtendedWBCDataset(
        dataset="JTSC", split="test", label=label, random_seed=random_seed
    )

    check_dataset_splits(data_support, data_calib, data_test)

    assert n_support_samples <= len(data_support), "Not enough support samples"

    support_images, support_labels = zip(
        *itertools.islice(data_support, n_support_samples)
    )
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.stack(support_labels).to(device)

    return data_support, data_calib, data_test, support_images, support_labels


def setup_oasis(label, n_support_samples, random_seed, device="cpu"):
    data_support = ExtendedOASISDataset(
        split="support", label=label, random_seed=random_seed
    )
    data_calib = ExtendedOASISDataset(
        split="calibration", label=label, random_seed=random_seed
    )
    data_test = ExtendedOASISDataset(split="test", label=label, random_seed=random_seed)

    check_dataset_splits(data_support, data_calib, data_test)

    assert n_support_samples <= len(data_support), "Not enough support samples"

    support_images, support_labels = zip(
        *itertools.islice(data_support, n_support_samples)
    )
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.stack(support_labels).to(device)

    return data_support, data_calib, data_test, support_images, support_labels


def download_repository(repo_url: str, download_path: str):
    """
    Clone a Git repository to the specified download path if it does not already exist.

    Args:
        repo_url (str): The URL of the Git repository to clone.
        download_path (str): The local path where the repository should be cloned.

    Returns:
        pathlib.Path: The path to the downloaded repository.
    """
    destination = pathlib.Path(download_path)

    if not destination.exists():
        subprocess.run(
            ["git", "clone", repo_url, str(destination)],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return destination


@dataclass
class ExtendedWBCDataset(WBCDataset):
    """
    Extends WBCDataset to allow for splitting the data into support, calibration, and test sets.
    Calibration data is used to compute conformal prediction scores and losses.

    source: https://github.com/JJGO/UniverSeg/blob/833a0c34c65e38d675e21bd48ddec6797cc03259/example_data/wbc.py#L55
    """

    ## === additional code: to enable further splitting for conformal pred ===
    calib_test_ratio: float = 0.5
    random_seed: Optional[int] = None

    def _split_indexes(self):
        # print(f" --- splitting data into {self.split} set")
        num_samples = len(self._data)
        np.random.seed(self.random_seed)
        permuted_indices = np.array([i for i in range(num_samples)])
        np.random.shuffle(permuted_indices)

        support_end_index = int(np.floor(self.support_frac * num_samples))

        # calibration data is taken from partitioning test data: it should represent "production data"
        num_calibration_samples = int(np.floor(self.calib_test_ratio * 100))

        self.support_indices = permuted_indices[:support_end_index]
        self.calibration_indices = permuted_indices[
            support_end_index : support_end_index + num_calibration_samples
        ]
        self.test_indices = permuted_indices[
            support_end_index + num_calibration_samples :
        ]

        assert set(self.support_indices).isdisjoint(
            set(self.calibration_indices)
        ), "Support and calibration sets are not disjoint"
        assert set(self.support_indices).isdisjoint(
            set(self.test_indices)
        ), "Support and test sets are not disjoint"
        assert set(self.calibration_indices).isdisjoint(
            set(self.test_indices)
        ), "Calibration and test sets are not disjoint"

        return {
            "support": permuted_indices[self.support_indices],
            "calibration": permuted_indices[self.calibration_indices],
            "test": permuted_indices[self.test_indices],
        }[self.split]


@dataclass
class ExtendedOASISDataset(OASISDataset):
    """
    Extends OASISDataset to allow for splitting the data into support, calibration, and test sets.
    Calibration data is used to compute conformal prediction scores and losses.

    source:
    - https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md
    - https://github.com/JJGO/UniverSeg/blob/833a0c34c65e38d675e21bd48ddec6797cc03259/example_data/oasis.py#L71
    """
    calib_test_ratio: float = 0.5
    random_seed: Optional[int] = None

    def _split_indexes(self):
        # print(f" --- splitting data into {self.split} set")
        num_samples = len(self._data)
        np.random.seed(self.random_seed)
        permuted_indices = np.array([i for i in range(num_samples)])
        np.random.shuffle(permuted_indices)

        support_end_index = int(np.floor(self.support_frac * num_samples))

        # calibration data is taken from partitioning test data: it should represent "production data"
        num_calibration_samples = int(np.floor(self.calib_test_ratio * 100))

        self.support_indices = permuted_indices[:support_end_index]
        self.calibration_indices = permuted_indices[
            support_end_index : support_end_index + num_calibration_samples
        ]
        self.test_indices = permuted_indices[
            support_end_index + num_calibration_samples :
        ]

        assert set(self.support_indices).isdisjoint(
            set(self.calibration_indices)
        ), "Support and calibration sets are not disjoint"
        assert set(self.support_indices).isdisjoint(
            set(self.test_indices)
        ), "Support and test sets are not disjoint"
        assert set(self.calibration_indices).isdisjoint(
            set(self.test_indices)
        ), "Calibration and test sets are not disjoint"

        return {
            "support": permuted_indices[self.support_indices],
            "calibration": permuted_indices[self.calibration_indices],
            "test": permuted_indices[self.test_indices],
        }[self.split]


# Create directories if they don't exist
# Save all predictions in a single npz file
def make_universeg_predictions(
    dataset,
    dataset_name,
    support_images,
    support_labels,
    device_str,
    universeg_model,
    save_to: Optional[str],
    verbose: bool = False,
):
    images = []
    gt_masks = []
    preds = []

    support_images = rearrange(support_images, "b c h w -> 1 b c h w")
    support_labels = rearrange(support_labels, "b c h w -> 1 b c h w")

    for image, gt_mask in tqdm(
        dataset, desc=f"Processing {dataset_name}", disable=not verbose
    ):
        image = rearrange(image, "c h w -> 1 c h w").to(device_str)
        gt_mask = gt_mask.to(device_str)

        logits = universeg_model(
            image,
            support_images,
            support_labels,
        )[0]

        pred = torch.sigmoid(logits).detach().cpu().numpy()
        images.append(image[0].cpu().numpy())
        gt_masks.append(gt_mask.cpu().numpy())
        preds.append(pred)

    if save_to is not None:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        dir_path_to_preds = f"{save_to}"
        os.makedirs(dir_path_to_preds, exist_ok=True)

        destination_file = f"{dir_path_to_preds}/{dataset_name}.npz"

        print(f" --- saving predictions to: {destination_file}")

        np.savez_compressed(
            file=destination_file,
            images=np.array(images),
            gt_masks=np.array(gt_masks),
            preds=np.array(preds),
        )

    return images, gt_masks, preds
