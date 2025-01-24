# EgoHand: Ego-centric Hand Pose Estimation and Gesture Recognition with Head-mounted Millimeter-wave Radar and IMUs
Recent advanced Virtual Reality (VR) headsets, such as the Apple Vision Pro, employ bottom-facing cameras to detect hand gestures and inputs, which offers users significant convenience in VR interactions. However, these bottomfacing cameras can sometimes be inconvenient and pose a risk of unintentionally exposing sensitive information, such as private body parts or personal surroundings. To mitigate these issues, we introduce EgoHand. This system provides an alternative solution by integrating millimeter-wave radar and IMUs for hand gesture recognition, thereby offering users an additional option for gesture interaction that enhances privacy protection. To accurately recognize hand gestures, we devise a two-stage skeleton-based gesture recognition scheme. In the first stage, a novel end-to-end Transformer architecture is employed to estimate the coordinates of hand joints. Subsequently, these estimated joint coordinates are utilized for gesture recognition. Extensive experiments involving 10 subjects show that EgoHand can detect hand gestures with 90.8% accuracy. Furthermore, EgoHand demonstrates robust performance across a variety of cross-domain tests, including different users, dominant hands, body postures, and scenes.


## Prerequisites

- Linux
- Python 3.7
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/WhisperYi/mmVR.git
cd mmVR
```

- Install [PyTorch](http://pytorch.org) and other dependencies (e.g., torchvision, torch, numpy).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### mmVR dataset

- Download [mmVR_dataset](https://kaggle.com/datasets/cdf079d9f49052500a08482b692eac6758d83e6b1ba9868d677c5cb814c427aa):
  - Download the `dataset.zip`, unzip it and move it to `./data/`

- Train and test model by mmwave + imu (stage1):

```bash
python train_kpt.py 
```

- Train and test model by keypoint (stage2):

```bash
python train_cls.py 
```

### File Structrue
```bash
.
│  config.py
│  train_cls.py
│  train_kpt.py
│  requirements.txt
│  environment.yaml
│  
├─data
│  │  eval_list.txt
│  │  train_list.txt
│  │  
│  ├─imu
│  │      XX_XX_XX_XX.npy
│  │      
│  ├─kpt_gt
│  │      XX_XX_XX_XX.npy
│  │      
│  ├─kpt_output
│  │      XX_XX_XX_XX.npy
│  │      
│  └─mmwave
│          XX_XX_XX_XX.mat
│          
├─dataset
│      datasets.py
│      dataset_kpt.py
│      
├─experiments
│  ├─conf_matrix
│  ├─param
│  ├─savept
│  └─weights
├─logs
├─models
│      backbone.py
│      mmVR_Transformer.py
│      position_encoding.py
│      ResNet.py
│      Transformer_layers.py
│      
└─utils
        loss.py
        matcher.py
        misc.py
```
