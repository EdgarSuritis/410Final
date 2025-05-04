# YOLO‑v11 — SIXray Dangerous‑Object Detector

This sub‑directory contains the code, data‑prep scripts, configuration files and results that reproduce our **YOLO‑v11** experiments from the poster.

> • Framework : [Ultralytics YOLO v0.4+](https://docs.ultralytics.com/)  
> • Backbone : `yolov11l.pt` (53 M params)  
> • Input size : 416 × 416  
> • Metrics    : mAP50 = 0.737, overall F1 ≈ 0.70 on SIXray 5‑class test split  

---

## 1 Expected Directory layout

```text
YOLO-v11/
├─ data/                     # created automatically
│   ├─ images/{train,val,test}/
│   └─ labels/{train,val,test}/
├─ runs/                     # output of ultralytics (ignored by Git)
│   └─ yolov11/phase{1,2}/...
├─ training-results/         # static copy of best run (for quick inspection)
│   ├─ results.png
│   ├─ PR_curve.png
│   ├─ R_curve.png
│   ├─ weights/
│   │   └─ best.pt           # same as runs/.../weights/best.pt
│   └─ *.jpg                 # sample batches + label visualisations
├─ scripts/
│   ├─ 00_make_splits.py     # generate train/val/test CSVs
│   ├─ 01_xml2yolo.py        # convert SIXray XML → YOLO .txt boxes
│   └─ 02_train_yolov11.py   # CLI wrapper around Ultralytics API
├─ data.yaml                 # YOLO dataset config (paths + names)
└─ README.md                 # you are here
```

> **Tip :** Keep `runs/` and large checkpoint files under **Git LFS** or add them to `.gitignore`. Otherwise, github will have issues.

---

## 2 Quick‑start (replicate our results)

### 0 Prerequisites

1. Follow the top‑level `README.md` up to **“Clone & set‑up”**  
   (*You should end up in the `xray-detection` conda environment with all dependencies installed.*)

Since we're working in a python notebook, you should choose the environment as your notebook's kernel.

2. Install the **YOLO CLI** if you skipped it in the main environment:
   ```bash
   pip install ultralytics
   ```

### 1 Download SIXRay

After extraction of the SIXRay dataset, you should have:

```
dataset/
   ├─ JPEGImages/ # over 1M images
   └─ positive-Annotation/ # positive XML annotations
```

### 2 Create splits & YOLO labels

The first cell in `YOLO-v11.ipynb` script builds the data-split .csv files and writes `data.yaml` with the correct relative paths.

### 3 Train YOLO‑v11 (2‑phase schedule)

Follow the python notebook file 

Training logs and TensorBoard files live under `runs/detect/yolov11/*`.

### 4 Evaluate

Outputs (P, R, mAP50, etc.) are printed to console and saved to
`.../runs/detect/val/results.csv`.  
Key plots (`PR_curve.png`, `labels.jpg`, etc.) are copied to the same directory folder for convenience.

---

## 4 Re‑using the model

Find the file with the weights and use your code to pull them for model re-use. They should be under: `.../runs/yolo11/phase2/weights/best.pt`

---

## 5 Troubleshooting

* **CUDA out of memory** → reduce `--batch` or `imgsz`; 16 GB GPU fits batch 16 @ 416².  
* **`FileNotFoundError` for XML** → check that `Annotations/` is at the same directory depth as `JPEGImages/`. 

---

## 6 References
 
* Ultralytics YOLO Docs <https://docs.ultralytics.com/>  
