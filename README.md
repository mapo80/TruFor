# TruFor

[![TruFor](https://img.shields.io/badge/TruFor%20webpage-222222.svg?style=for-the-badge&logo=github)](https://grip-unina.github.io/TruFor)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://doi.org/10.48550/arXiv.2212.10957)
[![GRIP](https://img.shields.io/badge/-GRIP-0888ef.svg?style=for-the-badge)](https://www.grip.unina.it)

Official PyTorch implementation of the paper "TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization"

<p align="center">
 <img src="./docs/teaser.png" alt="teaser" width="70%" />
</p>

## News
*   TODO: release Noiseprint++ training code
*   2025-03-05: Training code is now available
*   2023-06-28: Test code is now available
*   2023-02-27: Paper has been accepted at CVPR 2023
*   2022-12-21: Paper has been uploaded on arXiv


## Overview

**TruFor** is a forensic framework that can be applied to a large variety of image manipulation methods, from classic cheapfakes to more recent manipulations based on deep learning. We rely on the extraction of both high-level and low-level traces through a transformer-based fusion architecture that combines the RGB image and a learned noise-sensitive fingerprint. The latter learns to embed the artifacts related to the camera internal and external processing by training only on real data in a self-supervised manner. Forgeries are detected as deviations from the expected regular pattern that characterizes each pristine image. Looking for anomalies makes the approach able to robustly detect a variety of local manipulations, ensuring generalization. In addition to a pixel-level **localization map** and a whole-image **integrity score**, our approach outputs a **reliability map** that highlights areas where localization predictions may be error-prone. This is particularly important in forensic applications in order to reduce false alarms and allow for a large scale analysis. Extensive experiments on several datasets show that our method is able to reliably detect and localize both cheapfakes and deepfakes manipulations outperforming state-of-the-art works.


## Architecture

<center> <img src="./docs/architecture.png" alt="architecture" width="80%" /> </center>

We cast the forgery localization task as a supervised binary segmentation problem, combining high-level (**RGB**) and low-level (**Noiseprint++**) features using a cross-modal framework.


## Docker Setup (inference only)

Follow the instructions in the README.md in the `test_docker` folder.

## Training and inference

Follow the instructions in the README.md in the `TruFor_train_test` folder.


## CocoGlide dataset

You can download the CocoGlide dataset [here](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip).

### Sample subset

A subset of 50 images (25 real and 25 fake) from CocoGlide is provided in
`sample_dataset/` for quick tests. To recreate it from the full dataset:

```bash
wget https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip
unzip CocoGlide.zip
mkdir -p sample_dataset/real sample_dataset/fake
ls CocoGlide/real | head -n 25 | xargs -I{} cp CocoGlide/real/{} sample_dataset/real/
ls CocoGlide/fake | head -n 25 | xargs -I{} cp CocoGlide/fake/{} sample_dataset/fake/
```

## ONNX conversion

Install the required packages (CPU versions of PyTorch and torchvision are
recommended):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu -r onnx_requirements.txt
```

Export the pretrained models to ONNX:

```bash
PYTHONPATH=. python scripts/export_to_onnx.py --output onnx_models
```

The script generates `onnx_models/noiseprint_pp.onnx` (included in the
repository) and `onnx_models/mit_b2.onnx` (≈150 MB). Because of its size,
`mit_b2.onnx` is stored in this repository as multiple chunks; see the
`ONNX split/merge` section below for instructions on rebuilding the full
file.

## ONNX vs. PyTorch comparison

The script `scripts/test_onnx_models.py` compares the outputs of the
ONNX models with the original PyTorch implementations on the sample
subset. It prints per-image mean absolute errors and summary statistics.
For a faster run you can limit the number of processed images:

```bash
PYTHONPATH=. python scripts/test_onnx_models.py --max-images 5
```

Example output on the first five images:

```
image   noiseprint_mae  mit_b2_mae
glide_inpainting_val2017_100582_up.png  2.703454e-05    3.824630e+00
glide_inpainting_val2017_101762_up.png  2.865557e-05    3.819625e+00
glide_inpainting_val2017_10363_up.png   2.555746e-05    3.817635e+00
glide_inpainting_val2017_104198_up.png  5.236697e-05    3.835972e+00
glide_inpainting_val2017_104612_up.png  2.349462e-05    3.760577e+00

Noiseprint++ MAE: 3.142183e-05 (std 1.06e-05, max 5.24e-05)
mit_b2 MAE: 3.811688e+00 (std 2.63e-02, max 3.84e+00)
```

Running on all 50 images yields the following average absolute errors:

| Model        | MAE |
|--------------|------------------|
| Noiseprint++ | 2.63 × 10⁻⁵ |
| mit_b2       | 3.74 |

## TruFor.Sdk (.NET 9)

The repository includes a simple .NET 9 SDK that performs inference with
the ONNX models using [SkiaSharp](https://github.com/mono/SkiaSharp) for
image loading and [OnnxRuntime](https://onnxruntime.ai/) for model
execution.

Install the required .NET version using the provided script:

```bash
chmod +x dotnet-install.sh
./dotnet-install.sh -c 9.0
```

Copy the ONNX models into `TruFor.Sdk/models` so they are available at
runtime:

```bash
mkdir -p TruFor.Sdk/models
cp onnx_models/noiseprint_pp.onnx TruFor.Sdk/models/
./onnx_chunk.sh merge onnx_models/mit_b2 TruFor.Sdk/models/mit_b2.onnx
```

The SDK looks for these files in its `models` directory and does not
perform any merging at runtime.

Build the SDK and the sample console application:

```bash
~/.dotnet/dotnet build TruFor.Sdk/TruFor.Sdk.csproj
~/.dotnet/dotnet build TruFor.Sdk.Sample/TruFor.Sdk.Sample.csproj
```

Run inference on an image; the sample app writes intermediate results as
JSON files:

```bash
~/.dotnet/dotnet run --project TruFor.Sdk.Sample/TruFor.Sdk.Sample.csproj \
    sample_dataset/real/airplane_139871.png dotnet_out
```

### Comparison with PyTorch and ONNX

For the image above we compared the outputs of the SDK with the original
PyTorch models and with ONNXRuntime (Python). The mean absolute error
(MAE) shows that the .NET implementation matches the ONNX results while
deviating from the PyTorch outputs by the same amount:

| Output | .NET vs PyTorch MAE | .NET vs ONNX MAE |
|-------|--------------------:|-----------------:|
| Noiseprint | 2.53 × 10⁻⁵ | 0 |
| Seg0 | 3.89 | 7.4 × 10⁻⁴ |
| Seg1 | 3.49 | 4.4 × 10⁻⁴ |
| Seg2 | 3.77 | 2.9 × 10⁻⁴ |
| Seg3 | 3.75 | 2.4 × 10⁻⁴ |

## ONNX split/merge

The helper script `onnx_chunk.sh` splits a large ONNX file into smaller
chunks so it can be tracked by git and later reassembled when needed.

Make the script executable:

```bash
chmod +x onnx_chunk.sh
```

### Split

Split the original model into multiple files (e.g., `onnx_models/mit_b2_00`,
`onnx_models/mit_b2_01`, ...):

```bash
./onnx_chunk.sh split onnx_models/mit_b2.onnx onnx_models/mit_b2
```

Only commit the generated chunk files to the repository.

### Merge

Recreate the full `mit_b2.onnx` file from the chunks:

```bash
./onnx_chunk.sh merge onnx_models/mit_b2 onnx_models/mit_b2.onnx
```

Use the merged file locally and avoid committing it back to the repository.

## License

Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA'). 

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 


## Bibtex
 
 ```
 @InProceedings{Guillaro_2023_CVPR,
    author    = {Guillaro, Fabrizio and Cozzolino, Davide and Sud, Avneesh and Dufour, Nicholas and Verdoliva, Luisa},
    title     = {TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20606-20615}
}
```


## Acknowledgments
 
We gratefully acknowledge the support of this research by the Defense Advanced Research Projects Agency (DARPA) under agreement number FA8750-20-2-1004. 
The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon.
The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of DARPA or the U.S. Government.

In addition, this work has received funding by the European Union under the Horizon Europe vera.ai project, Grant Agreement number 101070093, and is supported by Google and by the PREMIER project, funded by the Italian Ministry of Education, University, and Research within the PRIN 2017 program.
Finally, we would like to thank Chris Bregler for useful discussions and support.
