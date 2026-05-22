<p align="center">
  <h1 align="center"><i>HOLO</i>: Homography-Guided Pose Estimator Network for Fine-Grained Visual Localization on SD Maps</h1>
  <h3 align="center">CVPR 2026</h3>
  <p align="center">
                <span class="author-block">
                <a href="" target="_blank">Xuchang Zhong</a><sup>1</sup>,
              </span>
              <span class="author-block">
                <a href="" target="_blank">Xu Cao</a><sup>1</sup>,
              </span>
              <span class="author-block">
                <a href="" target="_blank">Jinke Feng</a><sup>2</sup>,
              </span>
              <span class="author-block">
                <a href="" target="_blank">Hao Fang</a><sup>1,§</sup>
              </span>
  </p>

  <p align="center">
    <sup>1</sup> Beijing Institute of Technology &nbsp;&nbsp;
    <sup>2</sup> University of Science and Technology of China &nbsp;&nbsp;
    <sup>§</sup> Corresponding Author
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2601.02730"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2601.02730-b31b1b.svg"></a>
    <a href="https://drive.google.com/drive/folders/1Knq3lZxJr6QPe8qE9mIzaw57T7PH36xt?usp=sharing"><img alt="Google Drive" src="https://img.shields.io/badge/Google%20Drive-Download-4285F4?logo=googledrive&logoColor=white"></a>
  </p>

  <div align="center">
    <img src="./assets/teaser.png", width="800">
    <p align="left">
</p>
  </div>
</p>

##

This is the official repository of HOLO. HOLO is a multi-view pose estimation network for visual localization on SD maps. To address the inefficiency and unstable optimization of existing regression-based methods caused by the lack of explicit geometric guidance and unconstrained pose regression, HOLO introduces homography-based geometric priors to guide feature alignment and impose geometric constraints on pose estimation.

## 🚀 Getting Started
> Code is coming soon!
### Installation
First, clone the repository and create a data directory:

```bash
git clone https://github.com/Quartararo0714/HOLO
cd HOLO
```
Then, create conda environment with python 3.10 and intall the packages
```bash
conda create -n holo python=3.10
conda activate holo
pip install -r requirements.txt
```
To run the code with CUDA properly, you can comment out `torch` and `torchvision` in `requirement.txt`, and install the appropriate version of `torch>=2.1.0+cu121` and `torchvision>=0.16.0+cu121` according to the instructions on [PyTorch](https://pytorch.org/get-started/locally/).

### Data Preparation

### Inference

### Training

## To Do List
- \[ ] Code for training
- \[ ] Weights of model
- \[ ] Code for validation
- \[ ] Tutorial for installation
- \[ ] The osm files of SD maps
- \[x] Initial repo & main paper

## ❤️ Ackowledgement

## 📌 Citation
If you find our work useful, please consider giving us a star &#127775; and citing our paper with the following BibTeX entry.

