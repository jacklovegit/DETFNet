# DETF-Net: A Network for Retinal Vessel Segmentation Utilizing Detailed Feature Enhancement and Dynamic Temporal Fusion

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jacklovegit/DETFNet.git
   ```

2. Set up the MMSegmentation environment: Follow the official installation instructions for [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

## Usage

### Training

To train DETF-Net, run the following command:

```bash
python train.py work_dirs/unet-s5-d16_deeplabv3_4xb4-40k_stare-128x128/deeplabv3_stare-128x128-unet.py
```

### Inference

For running inference, use the following script:

```bash
python tools/test.py work_dirs/unet-s5-d16_deeplabv3_4xb4-40k_stare-128x128/deeplabv3_stare-128x128-unet.py [your-checkpoint-file].pth --eval mIoU
```

## Acknowledgments

This model is built upon the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) framework. We thank the authors of MMSegmentation for their incredible work.

