# Vision Transformer for Multi-Class X-Ray Weapon Detection

This project fine-tunes a pretrained Vision Transformer (ViT-B/16) to perform **multi-label classification** on X-ray baggage scans from the SIXray dataset. It classifies each image for the presence of any of five dangerous objects: **knife**, **gun**, **wrench**, **pliers**, and **scissors**.

Similar to the other models in our project, our ViT model directly uses the raw X-ray images and corresponding XML annotations to learn per-image object presence through **multi-label supervised training**.

---

## Project Highlights

- **Model**: Vision Transformer (ViT-B/16, pretrained on ImageNet)
- **Dataset**: SIXray (positive annotations only)
- **Input**: JPEG X-ray images + XML object annotations
- **Labels**: Multi-hot vectors for 5 dangerous object classes
- **Evaluation**: Accuracy, precision, recall, F1 score, and confusion matrix (visualized)
- **Performance**: Achieves high per-class F1 and robust generalization on validation split

---

## Setup & Dependencies

### 1. Clone the repository

```bash
git clone https://github.com/EdgarSuritis/410Final.git
cd ViT_model
```

### 2. Install requirements

Ensure you are in a clean Python environment (e.g. `venv` or `conda`), then run:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

We use the **positive annotations** subset of the [SIXray dataset](https://github.com/MeioJane/SIXray), which contains only images with dangerous items and their bounding box annotations.

### 1. Please follow the instructions in the README.md file in the main folder. If you have already done this for another model, you may skip this step.

- After extraction, ensure the directory contains:

```
📁 JPEGImages/
    P00001.jpg
    P00002.jpg
    ...
📁 positive-Annotation/
    P00001.xml
    P00002.xml
    ...
```

> ⚠️ Do **not** rename these folders. Then load the folders as directed in vit_xray_classification.py

---

## Training & Evaluation

To train and evaluate the ViT model on the dataset:

```bash
python vit_xray_classification.py
```

This script will:
- Automatically split the data 80/20 into training and validation
- Fine-tune the pretrained ViT on your data
- Report:
  - Pre- and post-fine-tune accuracy
  - Per-class precision, recall, and F1
  - Micro, macro, and weighted F1 scores
  - A 2x2 confusion matrix visualized as a heat map

> All outputs are printed to console and a confusion matrix is displayed at the end.

---

## Interpretation

Our model demonstrates that ViTs, even pretrained on natural images, can be successfully fine-tuned to detect subtle radiographic features of weapons in X-ray scans. High per-class metrics confirm its ability to distinguish between different object types under occlusion and low contrast.

---

## Files

| File                      | Description                                        |
|---------------------------|----------------------------------------------------|
| `vit_xray_classification.py` | Main training and evaluation script                |
| `requirements.txt`        | All Python dependencies for replication            |
| `README.md`               | This documentation                                 |



---


