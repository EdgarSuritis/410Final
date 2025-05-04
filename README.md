# Dangerous Object Detection in X‑ray Images (SIXray)

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Overview
Airport and public‑transit security still relies heavily on human X‑ray baggage screening, where overlapping objects, low‑contrast imagery and arbitrary orientations make dangerous items (knives, firearms, scissors, etc.) difficult to spot.  
Our project fine‑tunes state‑of‑the‑art vision backbones originally trained on natural images and benchmarks their effectiveness on a 16 k‑image subset of the public **SIXray** dataset.

- **ViT‑b16** reached a micro‑F1 = **0.84** on five weapon classes, outperforming a **ResNet‑50** baseline (**F1 = 0.80**) and a **YOLO‑v11** detector (**mAP50 = 0.737, F1 ≈ 0.70**).  
- Qualitative bounding‑box visualisations show that the model correctly localises partially occluded items at real‑time inference speeds on a single Apple M4 chip and GPU.

These results indicate that transformer backbones can overcome low‑contrast and overlap challenges, and serve as practical decision‑support tools for baggage officers.

---

## Replication Instructions

> Tested on Ubuntu 22.04 and macOS 13 with Python 3.10 and CUDA 12.

### 0. Prerequisites
- Git ≥ 2.40, Conda ≥ 23.3 (or pip).
- If using GPUs: NVIDIA driver ≥ 535 and CUDA‑enabled PyTorch.

### 1. Clone & set‑up
- Use the specific model implementation's package setup.
```bash
git clone https://github.com/EdgarSuritis/410Final.git
cd xray-detection
conda env create -f environment.yaml         # or: pip install -r requirements.txt
conda activate xray-detection
```

### 2. Download data
```bash
for i in $(seq -w 01 10); do                                         
  wget "http://aivc.ks3-cn-beijing.ksyun.com/data/public_data/SIXray-rar/dataset.part$i.rar";
done

```
The script pulls the SIXray dataset from the original data owner found [here](https://github.com/MeioJane/SIXray)

### 3. Train & fine‑tune & evaluate
1. [**ViT-b16**](https://github.com/EdgarSuritis/410Final/tree/main/ViT_model)
2. [**ResNet-50**](https://github.com/EdgarSuritis/410Final/tree/main/ResNet-50)
3. **YOLO-v11:** 
@TODO (Point to the specific folders for each of the models)

---

## Project Structure
@TODO (Print out the project structure in ASCII structure. Just the final file structure we have.)

---

## Future Directions
1. **Meta‑learning for rapid adaptation** – incorporating few‑shot fine‑tuning to handle emerging threats without relabelling thousands of images.  
2. **Domain‑robust augmentation** – leveraging stochastic geometric transforms to desensitise the model to specific scanner configurations and reduce dataset specificity.

---

## Contributions
| Member | GitHub ID | Primary tasks | Time Spent |
|--------|-----------|---------------|------------|
| Edgar Suritis | [@EdgarSuritis](https://github.com/EdgarSuritis) | ResNet baseline, demo app, environment setup | 30 Hours |
| Rashik Azad   | [@rashikcolgate](https://github.com/rashikcolgate) | Data pipeline, ViT training, poster layout | 30 Hours |
| James Njoroge | [@James-Njoroge](https://github.com/James-Njoroge) | YOLO‑v11 experiments, evaluation scripts, README | 30 Hours |

Find our poster [here](https://github.com/EdgarSuritis/410Final/blob/main/Poster.pdf).
---

## Citation
```bibtex
@misc{suritis2025xray,
  title        = {Dangerous Object Detection in X-ray Images},
  author       = {Suritis, Edgar and Azad, Rashik and Njoroge, James},
  year         = {2025},
  howpublished = {\url{https://github.com/EdgarSuritis/410Final}}
}
```

---

## License
This repository is released under the MIT License (see `LICENSE`).  
SIXray images remain © their authors and are recommended for research use only.

---

## Acknowledgements
- SIXray creators for the dataset.  
- Ultralytics YOLO and Timm maintainers for open‑source backbones.  
- [Prof. Forrest Davis](https://forrestdavis.github.io/) for his guidance and patience throughout our project.
