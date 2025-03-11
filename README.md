# ConSeMa: conformalization of segmentation by margin expansion

Code repository for the [paper](https://arxiv.org/abs/2503.05618):
> Mossina L. & Friedrich L. (2025). _Conformal Prediction for Image Segmentation Using Morphological Prediction Sets_. arXiv preprint arXiv:2503.05618.

- Lab: [DEEL](https://www.deel.ai/), at [IRT Saint Exupéry](https://www.irt-saintexupery.com/about/), Toulouse, France.
- Lab's open source [software](https://github.com/deel-ai) and [papers](https://github.com/deel-ai-papers)


## Idea
Use morphological operations (dilation) to add a margin around a predicted (binary) segmentation mask, such that the ground-truth mask is covered with high probability via conformal prediction.

![Dilation Animation](figures/dilation_animation.gif)


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
