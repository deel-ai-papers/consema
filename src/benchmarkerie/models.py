import torch
import numpy as np


def dice_score(y_pred, y_true) -> float:
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)

    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2 * (y_pred * y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()


class PranetPolypsPrecomputedInferencer:

    def __init__(
        self,
        device,
        return_numpy=True,
    ) -> None:
        self.device = device
        self.return_numpy = return_numpy

    @torch.no_grad()
    def inference(self, image, label, precomputed_soft_prediction=None):
        if precomputed_soft_prediction is None:
            raise ValueError("precomputed_soft_prediction must be provided")
        if not isinstance(image, np.ndarray):
            raise ValueError("precomputed_soft_prediction must be a numpy array")

        soft_pred = precomputed_soft_prediction
        hard_pred = soft_pred.round().clip(0, 1)
        segmentation_score = dice_score(hard_pred, label)

        if not self.return_numpy:
            raise NotImplementedError("Only numpy output is supported")

        return {
            "Image": image,
            "Soft Prediction": soft_pred,
            "Prediction": hard_pred,
            "Ground Truth": label,
            "segmentation_score": segmentation_score,
        }


class UniversegInferenceWrap:
    def __init__(
        self, model, support_images, support_labels, device, return_numpy=False
    ) -> None:
        self.model = model
        self.support_images = support_images
        self.support_labels = support_labels
        self.device = device
        self.return_numpy = return_numpy

    @torch.no_grad()
    def inference(self, image, label, precomputed_soft_prediction=None):
        # if not isinstance(image, np.ndarray):
        #     image, label = image.to(self.device), label.to(self.device)
        # else:
        image, label = torch.tensor(image).to(self.device), torch.tensor(label).to(
            self.device
        )

        logits = self.model(
            image[None], self.support_images[None], self.support_labels[None]
        )[0]
        soft_pred = torch.sigmoid(logits)
        hard_pred = soft_pred.round().clip(0, 1)

        segmentation_score = dice_score(hard_pred, label)

        if self.return_numpy:
            return {
                "Image": (
                    image.cpu().numpy() if isinstance(image, torch.Tensor) else image
                ),
                "Soft Prediction": soft_pred.cpu().numpy(),
                "Prediction": hard_pred.cpu().numpy(),
                "Ground Truth": label.cpu().numpy(),
                "segmentation_score": segmentation_score,
            }

        return {
            "Image": image,
            "Soft Prediction": soft_pred,
            "Prediction": hard_pred,
            "Ground Truth": label,
            "segmentation_score": segmentation_score,
        }
