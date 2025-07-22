# N-Pixels: A Novel Grey-Box Adversarial Attack for CNNs

This repository contains the source code and experimental setup for the paper:

> **N-Pixels: a Novel Grey-Box Adversarial Attack for Fooling Convolutional Neural Networks**  
> Salvatore Della Torca, Valentina Casola, Simone Izzo  
> _Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing (SAC '25), Catania, Italy_  
> [[DOI:10.1145/3672608.3707944](https://doi.org/10.1145/3672608.3707944)]

## 🔍 Overview

**N-Pixels (NPs)** is a novel *grey-box* adversarial attack targeting CNNs in the **evasion scenario**. The attack:
- **Identifies the most important region** of the input image responsible for the CNN’s decision, by leveraging the architecture of the model (without accessing weights or gradients).
- Applies **minimal, deterministic perturbations** to a subset of pixels in that region to induce misclassification, while ensuring the image remains visually indistinguishable from the original.

Unlike black-box attacks, NPs is not random; unlike white-box attacks, it doesn't require full internal access. It thus strikes a realistic balance between efficacy and applicability.


## 🧪 Experiments

The attack was tested against:

| Model        | Dataset         | Accuracy | N-Pixels Success Rate |
|--------------|------------------|----------|------------------------|
| ResNet24     | CIFAR-10         | 86.39%   | 91.9%                  |
| ResNet50     | Dogs vs Cats     | 97.01%   | 22.01% (↑ 40.09% if PSNR relaxed) |
| MobileNet-v3 | FGCV7 (plant leaves) | 98.08% | 89.17%                |

Perturbations are kept **visually imperceptible**, with PSNR mostly > 30dB.

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/<your-username>/n-pixels-attack.git
cd n-pixels-attack
pip install -r requirements.txt
```

### Prepare Datasets

Make sure to download and extract the datasets:
- **CIFAR-10**: automatically downloaded via torchvision
- **Dogs vs Cats**: [Kaggle link](https://www.kaggle.com/c/dogs-vs-cats)
- **FGCV7 (Apple Leaf)**: [Kaggle link](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)

Update paths in `datasets/config.py` if needed.

## 📌 How It Works

### Stage 1: Region Selection

- Uses the final feature map of the CNN to identify the most discriminative feature (fiber).
- Back-maps this feature through the convolutional layers to locate its origin in the input image.
- Applies K-means to extract a cluster of critical pixels for targeted perturbation.

### Stage 2: Perturbation

- Iteratively applies a minimal additive/subtractive RGB perturbation using binary search.
- Evaluates misclassification potential (𝑣) vs visual similarity (𝑤) using metrics like PSNR.
- Stops as soon as an adversarial sample is found within the similarity constraint.

## 📊 Evaluation

- Performance is measured in terms of **attack success rate** and **PSNR**.
- Baselines: compared against FGSM, PGD, BIM, SimBA, One-Pixel Attack.

## 📜 Citation

If you use this code or the N-Pixels attack in your research, please cite the following paper:

```bibtex
@inproceedings{10.1145/3672608.3707944,
  author = {Della Torca, Salvatore and Casola, Valentina and Izzo, Simone},
  title = {N-Pixels: a Novel Grey-Box Adversarial Attack for Fooling Convolutional Neural Networks},
  year = {2025},
  isbn = {9798400706295},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3672608.3707944},
  doi = {10.1145/3672608.3707944},
  booktitle = {Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing},
  pages = {1539–1547},
  numpages = {9},
  keywords = {convolutional neural networks, adversarial attacks, grey-box methodologies},
  location = {Catania International Airport, Catania, Italy},
  series = {SAC '25}
}
```

## 🙏 Acknowledgments

This work was carried out at the University of Napoli Federico II, with experiments performed in collaboration with the authors' research groups.
