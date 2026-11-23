# Third-party information and distribution review

This project source ZIP does not bundle third-party wheels, the TorchVision
pretrained checkpoint, or a training dataset. Direct dependencies are declared
as version ranges in `pyproject.toml`, not as a locked or exhaustive bill of
materials. Installed binary distributions can contain additional components and
licenses. Inspect the notices shipped with the exact versions you install and
satisfy their terms before redistributing binaries.

| Dependency | Upstream license information | Upstream source |
| --- | --- | --- |
| PyTorch | Main project BSD-3-Clause; distributions may contain components with additional terms. | https://github.com/pytorch/pytorch/blob/v2.10.0/LICENSE |
| TorchVision | Repository BSD-3-Clause; this does not by itself settle the terms of every pretrained model. | https://github.com/pytorch/vision/blob/v0.25.0/LICENSE |
| NumPy | BSD-3-Clause; review bundled components. | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| Pillow | MIT-CMU; binary builds may include other components. | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| OpenCV Python headless | Packaging scripts MIT, OpenCV Apache-2.0; wheels include third-party components with separate terms. | https://github.com/opencv/opencv-python#licensing |
| Gradio | Apache-2.0. | https://github.com/gradio-app/gradio/blob/main/LICENSE |
| Pytest (development) | MIT. | https://github.com/pytest-dev/pytest/blob/main/LICENSE |
| Ruff (development) | MIT. | https://github.com/astral-sh/ruff/blob/main/LICENSE |

The model is SqueezeNet 1.1 from TorchVision's `SqueezeNet1_1_Weights.DEFAULT`
(ImageNet-1K V1), used as a frozen encoder. Credit SqueezeNet to Iandola et al.,
*arXiv:1602.07360*; implementation and weights obtained via TorchVision:
https://docs.pytorch.org/vision/main/models/generated/torchvision.models.squeezenet1_1.html

TorchVision warns pretrained models may have separate license or dataset-derived
terms: https://docs.pytorch.org/vision/stable/models.html. The exact legal terms
for redistributing the SqueezeNet checkpoint were not established here; it is not
included. The experiment's separately packaged decoder was trained using local
photographs whose licenses and authorizations have not been established. Its
redistribution is not represented as cleared. Review training-data terms,
trained-weight distribution, and any institutional conditions before sharing the
decoder, even though the checkpoint ZIP contains no images.

No project LICENSE file was found. The project copyright holder must choose and
supply suitable project licensing or explicit submission permission. No project
license is assigned by this notice. The image inventory is documented separately
in `docs/submission/SUBMISSION_READINESS_AUDIT.md`.
