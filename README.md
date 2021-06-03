# Blind Super-Resolution using an Ensemble of Kernel-Specific Networks (EnsSR)

This repository is the official implementation of "Blind Super-Resolution using an Ensemble of Kernel-Specific Networks".

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
conda activate EnsSR_env
```

## Datasets and pretrained models

You can download all datasets (DIV2KRK, Flickr2KRK and Urban100RK) and pretrained models from the following links.
https://drive.google.com/drive/folders/1v7Lthkp-nLwdXGGkqolBgl5H88oNDH6l?usp=sharing

## Test on synthetic datasets

For example to evaluate on DIV2KRK dataset:

```eval
python main.py --in_dir ../datasets/DIV2KRK/lr_x2 --out_dir ../results/DIV2KRK/x2 --gt_dir ../datasets/DIV2KRK/gt --scale 2
python main.py --in_dir ../datasets/DIV2KRK/lr_x4 --out_dir ../results/DIV2KRK/x4 --gt_dir ../datasets/DIV2KRK/gt --scale 4
```

## Test on real images
To evaluate on real-world images:

```eval-dataset
python main.py --in_dir 'path to the LR input images' --out_dir 'path to save results' --scale 2 --real
python main.py --in_dir 'path to the LR input images' --out_dir 'path to save results' --scale 4 --real
```

## Results

Our model achieves the following performance values (PSNR / SSIM) on DIV2KRK, Urban100 and NTIRE2017 datasets:

| Model name         | DIV2KRK         | Urban100        | NTIRE2017        |
| ------------------ |---------------- |---------------- | ---------------- |
| DualSR             |  30.92 / 0.8728 |  25.04 / 0.7803 |  28.82 / 0.8045  |

## Acknowledgement

The code is built on [DualSR](https://github.com/memad73/DualSR) and [KernelGAN](https://github.com/sefibk/KernelGAN). We thank the authors for sharing the codes.
