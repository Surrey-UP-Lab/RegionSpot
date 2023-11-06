## Getting Started with RegionSpot


### Installation

The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2).

#### Requirements
- **Operating System**: Linux or macOS
- **Python**: Version 3.6 or newer
- **PyTorch**: Version 1.9.0 or newer, along with the compatible version of [torchvision](https://github.com/pytorch/vision/). You can install both from [pytorch.org](https://pytorch.org).


#### Steps
1. **Detectron2 Installation**:
   Install Detectron2 by following the official installation guide available here:
   [Detectron2 Installation Guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#installation).
1. **CLIP Installation**

    Install CLIP by following the official installation guide available here:
   [CLIP Installation](https://github.com/openai/CLIP).
2. **Data Preparation**:
   Organize your data according to the instructions provided in [DATA.md](./DATA.md) in this repository.

4. **Model Training**:
   To train the RegionSpot model, use the following command templates:

   ```bash
   # Stage 1 Training:
   python3 train_net.py --num-gpus 8 \
       --config-file configs/objects365_bl.yaml

   # Stage 2 Training:
   python3 train_net.py --num-gpus 8 \
       --config-file configs/objects365_v3det_openimages_bl.yaml

4. **Model Evaluation**:
   To evaluate the trained RegionSpot model, use the following command. Ensure that the `MODEL.CLIP_TYPE` and `MODEL.CLIP_INPUT_SIZE` corresponds to the particular `MODEL.WEIGHTS` you are using.

   ```bash
   python3 train_net.py --num-gpus 8 \
       --config-file configs/eval.yaml \
       --eval-only \
       MODEL.WEIGHTS /path/to/model_weights.pth \
       MODEL.CLIP_TYPE CLIP_400M_Large \
       MODEL.CLIP_INPUT_SIZE 224 
