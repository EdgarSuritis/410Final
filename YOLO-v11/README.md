# Dangerous Object Detection in X‑ray Images (SIXray)

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Overview
Airport and public‑transit security still relies heavily on human X‑ray baggage screening, where overlapping objects, low‑contrast imagery and arbitrary orientations make dangerous items (knives, firearms, scissors, etc.) difficult to spot.  
Our project fine‑tunes state‑of‑the‑art vision backbones originally trained on natural images and benchmarks their effectiveness on a 16 k‑image subset of the public **SIXray** dataset.

- **ViT‑b16** reached a micro‑F1 = **0.84** on five weapon classes, outperforming a ResNet‑50 baseline (F1 = 0.80) and a YOLO‑v11 detector (mAP50 = 0.737, F1 ≈ 0.70).  
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
git lfs install                             # one‑time (weights tracked via LFS)
git clone https://github.com/your-org/xray-detection.git
cd xray-detection
conda env create -f environment.yml         # or: pip install -r requirements.txt
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
| Member | GitHub ID | Primary tasks |
|--------|-----------|---------------|
| Edgar Suritis | [@EdgarSuritis](https://github.com/EdgarSuritis) | Data pipeline, ViT training, poster layout |
| Rashik Azad   | [@rashikcolgate](https://github.com/rashikcolgate) | YOLO‑v11 experiments, evaluation scripts, README |
| James Njoroge | [@James-Njoroge](https://github.com/James-Njoroge) | ResNet baseline, demo app, environment setup | 


---

## Citation
```bibtex
@misc{suritis2025xray,
  title        = {Dangerous Object Detection in X-ray Images},
  author       = {Suritis, Edgar and Azad, Rashik and Njoroge, James},
  year         = {2025},
  howpublished = {\url{https://github.com/your-org/xray-detection}}
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
