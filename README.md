# MoESR: Blind Super-Resolution using Kernel-Aware Mixture of Experts

This repository is the official implementation of "MoESR: Blind Super-Resolution using Kernel-Aware Mixture of Experts".

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
conda activate MoESR_env
```

## Datasets and pretrained models

You can download all datasets (DIV2KRK, Flickr2KRK and Urban100RK) and pretrained models from the following link:
https://drive.google.com/drive/folders/1v7Lthkp-nLwdXGGkqolBgl5H88oNDH6l?usp=sharing

## Test on synthetic datasets

For example to evaluate on DIV2KRK dataset:

```eval
cd codes
python main.py --in_dir ../datasets/DIV2KRK/lr_x2 --out_dir ../results/DIV2KRK/x2 --gt_dir ../datasets/DIV2KRK/gt --scale 2
python main.py --in_dir ../datasets/DIV2KRK/lr_x4 --out_dir ../results/DIV2KRK/x4 --gt_dir ../datasets/DIV2KRK/gt --scale 4
```

## Test on real images
To evaluate on real-world images:

```eval-dataset
cd codes
python main.py --in_dir 'path to the LR input images' --out_dir 'path to save results' --scale 2 --real
python main.py --in_dir 'path to the LR input images' --out_dir 'path to save results' --scale 4 --real
```

## Results

Our model achieves the following performance values (PSNR / SSIM) on DIV2KRK, Flickr2KRK and Urban100RK datasets:

| Model name         | Scale | DIV2KRK         | Flickr2KRK      | Urban100RK       |
| ------------------ |-------|---------------- |---------------- | ---------------- |
| MoESR              | x2    |  32.69 / 0.9054 |  32.95 / 0.9056 |  27.29 / 0.8448  |
| MoESR              | x4    |  28.48 / 0.7805 |  28.57 / 0.7795 |  23.62 / 0.6766  |

## Acknowledgement

The code is built on [DualSR](https://github.com/memad73/DualSR) and [KernelGAN](https://github.com/sefibk/KernelGAN). We thank the authors for sharing the codes.
