# Conformal Prediction for Image Segmentation Using Morphological Prediction Sets

**MICCAI 2025**. [📄 arXiv](https://arxiv.org/abs/2503.05618)

[Luca Mossina](https://scholar.google.com/citations?hl=en&user=SCpz8XMAAAAJ),¹ [Corentin Friedrich](https://scholar.google.com/citations?user=w6oH0xUAAAAJ&hl=en)¹

¹ [IRT Saint Exupéry](https://www.irt-saintexupery.com/smart-technologies/), Toulouse, France. 

- Research Lab: [DEEL](https://www.deel.ai), *Dependable, Explainable & Embeddable Learning* for trustworthy AI.
- Lab's open-source [software](https://github.com/deel-ai) and [papers](https://github.com/deel-ai-papers)


## Idea
We use [morphological operations](https://en.wikipedia.org/wiki/Mathematical_morphology) (dilation, sequences of dilations, etc.) to add a margin around a predicted (binary) segmentation mask, such that the ground-truth mask is covered with high probability via conformal prediction.

In the synthetic example below, the red pixels (bold contours) are false negatives, that is, they belong to the ground truth but were not predicted.
The animation shows **five sequential dilations** by a (3X3) cross structuring element, which expand the margin of the predicted mask (darker blue).
Three iterations is the minimal number of iterations needed, i.e. the _nonconformity score_: all missing pixels are recovered (shown in orange).

![Dilation Animation](figures/dilation_animation.gif)


## Examples
The directory [notebooks](/notebooks) contains complete examples for the datasets:
- [WBC](/notebooks/n201_consema_wbc.ipynb) and [OASIS](/notebooks/n202_consema_oasis.ipynb), using the _UniverSeg_ segmentation model
- [polyps](/notebooks/n203_consema_polyps.ipynb) tumors dataset, using _PraNet_ (we use precomputed predictions as distributed by [A. Angelopoulos](https://github.com/aangelopoulos/conformal-prediction/blob/67f506e4880e192ef9fc6a2de73e21b277f8c544/notebooks/tumor-segmentation.ipynb).


## References & sources
Starting points for datasets:
- [WBC](https://github.com/JJGO/UniverSeg/blob/833a0c34c65e38d675e21bd48ddec6797cc03259/example_data/wbc.py#L55)
- [OASIS](https://github.com/JJGO/UniverSeg/blob/833a0c34c65e38d675e21bd48ddec6797cc03259/example_data/oasis.py#L71) 
- [polyps tumor data](https://github.com/aangelopoulos/conformal-prediction/blob/67f506e4880e192ef9fc6a2de73e21b277f8c544/notebooks/tumor-segmentation.ipynb)

Models used:
- UniverSeg. [code](https://github.com/JJGO/UniverSeg), [paper](https://arxiv.org/abs/2304.06131)
- PraNet. [paper](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_26)


## Citation
```
@article{Mossina_2025_conformal,
  title={Conformal Prediction for Image Segmentation Using Morphological Prediction Sets},
  author={Mossina, Luca and Friedrich, Corentin},
  journal={arXiv preprint arXiv:2503.05618},
  year={2025}
}
```
