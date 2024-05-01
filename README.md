# mmVR: enhancing privacy in VR interactions with head-mounted millimeter-wave radar

Recent Virtual Reality (VR) headsets like Apple Vision Pro employ purely vision-based approaches for gesture recognition, offering great convenience to users for human-computer interaction. However, such solutions also heighten the risk of exposing sensitive information, such as private body parts or the faces of family members. To address this issue, we propose mmVR. By utilizing millimeter-wave radar as the gesture recognition component within VR devices, mmVR ensures enhanced privacy protection. To accurately recognize gestures, we devised a two-stage skeleton-based gesture recognition scheme. In the first stage, a novel end-to-end Transformer architecture is employed to estimate the positions of hand joints. Subsequently, in the second stage, these estimated joint positions are utilized for gesture recognition. Extensive experimental validation confirms that our two-stage approach significantly improves gesture recognition accuracy compared to single-stage methods, achieving an increase of over 30%.

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

- Download [mmVR_dataset](https://www.kaggle.com/xrfdataset/xrf55):
  - Download the `dataset.zip`, unzip it and move it to `./data/`

- Train and test model by mmwave + imu (stage1):

```bash
python train_kpt.py 
```

- Train and test model by keypoint (stage2)::

```bash
python train_cls.py 
```

### File Structrue
```bash
.
│  dml_eval.py
│  dml_train.py
│  environment.yaml
│  generate_txt.py
│  opts.py
│  README.md
│  requirements.txt
│  split_train_test.py
│  XRFDataset.py
├─dataset
│  ├─Raw_dataset
│  │  ├─mmWave
│  │  │      XX_XX_XX.npy
│  │  ├─RFID
│  │  │      XX_XX_XX.npy
│  │  └─WiFi
│  │          XX_XX_XX.npy
│  └─XRF_dataset
│      ├─test_data
│      │  ├─mmWave
│      │  │      XX_XX_XX.npy
│      │  ├─RFID
│      │  │      XX_XX_XX.npy
│      │  └─WiFi
│      │          XX_XX_XX.npy
│      └─train_data
│          ├─mmWave
│          │      XX_XX_XX.npy
│          ├─RFID
│          │      XX_XX_XX.npy
│          └─WiFi
│                  XX_XX_XX.npy  
├─model
│      resnet1d.py
│      resnet1d_rfid.py
│      resnet2d.py
├─result
│  ├─conf_matrix
│  ├─learning_curve
│  ├─params
│  └─weights
└─word2vec
        bert_new_sentence_large_uncased.npy
```
