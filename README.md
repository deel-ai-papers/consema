# ConSeMa: conformalization of segmentation by margin expansion

Code repository for the paper:
> L. Mossina & C. Friedrich (2025). _Conformal Prediction for Image Segmentation Using Morphological Prediction Sets_

- Lab: [DEEL](https://www.deel.ai/), at [IRT Saint Exupéry](https://www.irt-saintexupery.com/about/), Toulouse, France.
- Lab's open source [software](https://github.com/deel-ai) and [papers](https://github.com/deel-ai-papers)


## Idea
Use morphological operations (dilation) to add a margin around a predicted (binary) segmentation mask, such that the ground-truth mask is covered with high proability via conformal prediction.

![Dilation Animation](figures/dilation_animation.gif)


## References & sources
Starting points for datasets
- [WBC](https://github.com/JJGO/UniverSeg/blob/833a0c34c65e38d675e21bd48ddec6797cc03259/example_data/wbc.py#L55)
- [OASIS](https://github.com/JJGO/UniverSeg/blob/833a0c34c65e38d675e21bd48ddec6797cc03259/example_data/oasis.py#L71) 
- [polyps tumor data](https://github.com/aangelopoulos/conformal-prediction/blob/67f506e4880e192ef9fc6a2de73e21b277f8c544/notebooks/tumor-segmentation.ipynb)
