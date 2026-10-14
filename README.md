# Gao et al. (2020) pothole-method reproduction

This directory contains a Python/OpenCV reproduction of **Detection and
Segmentation of Cement Concrete Pavement Pothole Based on Image Processing
Technology**, DOI [`10.1155/2020/1360832`](https://doi.org/10.1155/2020/1360832).

Only the original paper is implemented here. The proposed texture-plus-edge
modification in `CV_ABSTRACT (1).pdf` is intentionally not included yet.

## What is reproduced

The implementation follows the paper's sequence:

1. RGB/BGR input and grayscale conversion.
2. 9x9 local Shannon-entropy texture filtering (GLCM/entropy concept in
   equations 1-3, implemented as the local entropy image used by MATLAB's
   `entropyfilt`).
3. Texture-image normalization and binarization.
4. Removal of connected objects smaller than 1,500 pixels.
5. Morphological closing and hole filling.
6. Extraction of the maximum connected domain and an image-mask overlay.
7. Nondefect screening at `A_sum < 0.03 A_total`.
8. The eight stated features: connected-domain count, projection ratio,
   box-counting fractal dimension, area, perimeter, circularity, rectangle
   degree, and aspect ratio.
9. A four-class C-SVC backed by LIBSVM (pothole, transverse crack,
   longitudinal crack, and complex crack), using the paper's 70/30 split.
10. Accuracy, pothole precision/recall/F1, paper overlap rate, IoU, and area
    extraction rate.

## Reproducibility boundary

The authors did not release their 610 classification images, 250 segmentation
images, trained model, source code, random split, entropy neighborhood,
threshold, or morphology element. Their data-availability statement says the
images are available from the corresponding author on request. Consequently,
this code can reproduce the published algorithm but cannot honestly reproduce
the paper's numerical results without those inputs.

The default 9x9 entropy neighborhood follows MATLAB `entropyfilt`; Otsu is used
when no threshold is supplied. The morphology element defaults to a 9x9
ellipse. Each unpublished parameter is exposed by the command line.

## Install and run

```powershell
python -m pip install -r requirements.txt
python -m paper_impl.cli segment path\to\image.jpg --output paper_output
```

The segment command writes every intermediate stage and prints the eight
features. If an in-frame ruler spans 420 pixels for 30 cm, equations (4)-(5)
can also report physical area:

```powershell
python -m paper_impl.cli segment image.jpg --scale-length-cm 30 --scale-length-pixels 420
```

To use a fixed normalized threshold, for example:

```powershell
python -m paper_impl.cli segment image.jpg --threshold 0.45 --close-kernel 9
```

Training data must have the paper's four labels as directory names:

```text
dataset/
  pothole/
  transverse_crack/
  longitudinal_crack/
  complex_crack/
```

Train and predict:

```powershell
python -m paper_impl.cli train dataset --model paper_model.joblib
python -m paper_impl.cli predict sample.jpg --model paper_model.joblib --output prediction
```

Evaluate a predicted binary mask against a manually labeled mask:

```powershell
python -m paper_impl.cli evaluate-mask predicted.png ground_truth.png
```

Run the included checks:

```powershell
python -m unittest discover -s tests -v
```
