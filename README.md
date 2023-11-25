# Conditional Diffusion Models for fMRI Denoising and Reconstruction

## Introduction
This project focuses on implementing a score-based diffusion model using PyTorch for the denoising and reconstruction of fMRI images. It aims to address the challenges in medical image processing, particularly in improving the clarity and quality of fMRI images by reducing Gaussian noise and reconstructing sparsely sampled single-channel MRI images.

## Features
- **Score-Based Diffusion Model:** Utilizes a diffusion model approach for effectively eliminating Gaussian noise from fMRI images.
- **High-Performance Computing:** Optimized for execution on high-performance compute clusters.
- **Euler-Maruyama Sampler:** Employed for achieving a Peak Signal-to-Noise Ratio (PSNR) of up to 22.7 in noise reduction.
- **Predictor-Corrector Sampler:** Modified for reconstructing sparsely sampled single-channel MRI images, achieving a PSNR of up to 39.

## Citation
To cite this work in your research, please use the following format:
```
@misc{conditional_diffusion_fmri,
author = {Madhav Khirwar},
title = {Conditional Diffusion Models for fMRI Denoising and Reconstruction},
year = {2022},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/madhavk98/STIR_Denoising_Diffusion}
}
```
MIT License

Copyright (c) 2022 [Madhav Khirwar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
