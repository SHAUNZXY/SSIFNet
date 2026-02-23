# SSIFNet

This repo is the official PyTorch implementation of **SSIFNet: Spatial-temporal stereo information fusion network for self-supervised surgical video inpainting**.

## Installation

Please follow the steps below to reproduce the experimental environment:

```bash
git clone https://github.com/SHAUNZXY/SSIFNet.git
cd SSIFNet

conda env create -f environment.yml
conda activate ssifnet
```

## Data Preparation

We use the publicly available [RIS-2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/) and [RSS-2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/) datasets from the EndoVis challenges. Please register and download the datasets from the official websites. Using the official camera calibration parameters provided with the datasets, we generate calibrated and rectified stereo image sequences for training and inference.

Taking **RIS-2017** as an example, please organize the dataset as follows:

```text
datasets/
└── ris2017/
    └── JPEGImages/
        ├── instrument_dataset_1_left.zip
        ├── instrument_dataset_1_right.zip
        ├── ...
        ├── instrument_dataset_8_left.zip
        └── instrument_dataset_8_right.zip
```

Each zip file includes all calibrated and rectified `.jpg` frames of either the left or the right view for one sequence. The provided `train.json` file specifies the sequence names and their corresponding frame counts used to construct training sequences.

## Training
Before training, please download the required pretrained models from [Google Drive](https://drive.google.com/drive/folders/1XTYJYJIeZ0K7KZ809OS0x9pUxK4WVZyu?usp=sharing) and place them under:
```text
load_model/
├── monodepth_pretrained.pth
└── i3d_rgb_imagenet.pt
```

Taking **RIS-2017** as an example, run the following command for training:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/ris2017.json
```

## Inference

To perform stereo video inpainting, please prepare:

- A pair of stereo videos (left and right views)
- Two corresponding folders of binary masks, one for each view.  
  Each mask folder should contain frame-wise binary mask images aligned with the video frames (one mask per frame).

You may download an example from [Google Drive](https://drive.google.com/drive/folders/1XTYJYJIeZ0K7KZ809OS0x9pUxK4WVZyu?usp=sharing) and organize the files as follows:

```text
examples/
├── instrument10test_orig_left.mp4
├── instrument10test_orig_right.mp4
├── instrument10test_orig_left/
└── instrument10test_orig_right/
```

Then run the following command to perform stereo video inpainting:

```bash
CUDA_VISIBLE_DEVICES=0 python inpaint_stereo_video.py --model ssifnet --video examples/instrument10test_orig_left.mp4 --mask examples/instrument10test_orig_left --ckpt <CKPT_PATH> --dilate 15
```

The official trained model `gen.pth` is available at [Google Drive](https://drive.google.com/drive/folders/1XTYJYJIeZ0K7KZ809OS0x9pUxK4WVZyu?usp=sharing).

## Reference

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{zou2025ssifnet,
  title={SSIFNet: Spatial--temporal stereo information fusion network for self-supervised surgical video inpainting},
  author={Zou, Xiaoyang and Zhang, Zhuyuan and Yu, Derong and Sun, Wenyuan and Liu, Wenyong and Hang, Donghua and Bao, Wei and Zheng, Guoyan},
  journal={Computerized Medical Imaging and Graphics},
  pages={102622},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement

We would like to thank the authors of the following open-source projects:

- [E2FGVI](https://github.com/MCG-NKU/E2FGVI)
- [STTN](https://github.com/researchmm/STTN)
- [FuseFormer](https://github.com/ruiliu-ai/FuseFormer)
- [Focal-Transformer](https://github.com/microsoft/Focal-Transformer)
- [Monodepth](https://github.com/OniroAI/MonoDepth-PyTorch)