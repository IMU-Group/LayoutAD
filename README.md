# [LayoutAD: Exploring Semantic-Geometric Misalignment Reasoning for Scene Layout Anomaly Detection]()

Official implementation of **LayoutAD**, from the following paper:

LayoutAD: Exploring Semantic-Geometric Misalignment Reasoning for Scene Layout Anomaly Detection, CVPR 2026

## Dataset
You can download our dataset at [[`Google Drive`](https://drive.google.com/file/d/1-YwE7tjcVjeM-qJ45E450mWq26e9wNct/view?usp=drive_link)]

## Installation
We suggest users to use the conda for creating new python environment.
- This project uses [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main) and [CLIP](https://github.com/openai/CLIP) for feature extraction.
- `pip install -r requirements.txt`

### Example conda environment setup
```bash
git clone https://github.com/IMU-Group/LayoutAD.git
cd LayoutAD
conda create -n layoutad python=3.10
conda activate layoutad

# Detectron2 & Mask2Former
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install 'git+https://github.com/facebookresearch/Mask2Former.git'
# CLIP
pip install 'git+https://github.com/openai/CLIP.git'

pip install -r requirements.txt
```

## Run Experiments


## Reference
