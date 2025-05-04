import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =============================================================================
# Dataset Class: Reads CSV file for file names and uses XML annotation files.
# =============================================================================
class SixrayDataset(Dataset):
    """
    Reads SIXray images and their annotations.
    
    It uses a CSV file that contains at least a column named "name" (e.g. "P00007")
    and (optionally) other columns (like "Gun", "Knife", etc.) which we ignore.
    
    For positive images (those whose name starts with "P"), it loads the corresponding
    XML file from the annotations directory; if the XML file is missing or the image is
    negative, it returns an empty array for bounding boxes.
    
    Each sample is a dictionary with:
      - "image": a NumPy array (RGB)
      - "bounding_boxes": a NumPy array of shape [N, 5] with rows:
              [class, x_center, y_center, w, h] (all normalized to [0,1])
    
    Args:
      csv_fname (str): Path to the CSV file (e.g., train.csv or test.csv).
      images_dir (str): Directory containing the JPEG images.
      annotations_dir (str): Directory containing the positive XML files.
      transform (callable, optional): A transform to be applied on the sample.
    """
    def __init__(self, csv_fname, images_dir, transform=None):
        self.fnames = pd.read_csv(csv_fname, delimiter=',', header=0)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # Get the image file name from CSV (assume the CSV has a column "name")
        row = self.fnames.iloc[idx]
        file_name = row['name']  # e.g., "P00007"
        
        image_path = os.path.join(self.images_dir, file_name + ".jpg")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Determine if image is positive by its name (if it starts with 'P', we assume positive)
        if file_name[0] == 'P':
            danger_category = 1 #c ontains a dangerous item
        else:
            danger_category = 0 # doesn't contain a dangerous item

        sample = {"image": image_np, "label": danger_category}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
# =============================================================================
# Transformation: We use Albumentations package to apply consistent augmentations to both image and boxes.
# =============================================================================
def custom_transform(sample, alb_transform):
    """
    Converts the sample dict into the format expected by Albumentations.
    
    - The sample contains "image" (as a NumPy array) and "bounding_boxes" (a NumPy array [N, 5]).
    - We convert the bounding_boxes into two lists:
         bboxes: list of [x_center, y_center, w, h]
         labels: list of class IDs.
    Then apply the alb_transform.
    """
    image = sample["image"]
    
    transformed = alb_transform(image=image)
    # Ensure the transformed image is a float tensor due to prior error.
    transformed["image"] = transformed["image"].float()
    return {"image": transformed["image"], "label": sample["label"]}
    
# =============================================================================
# A Wrapper Dataset that applies our custom_transform to the raw SixrayDataset output.
# =============================================================================
class SixrayWrapper(Dataset):
    def __init__(self, csv_fname, images_dir, alb_transform):
        self.dataset = SixrayDataset(csv_fname, images_dir, transform=None)
        self.alb_transform = alb_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]  # returns {"image": ..., "bounding_boxes": ..., "labels": ...}
        sample = custom_transform(sample, self.alb_transform)
        
        return sample["image"], sample["label"]

    
# =============================================================================
# Albumentations Transformation Pipelines for Training and Testing. (From Paper)
# =============================================================================
def get_albumentations_transform(train=True):
    if train:
        return A.Compose([
            A.Resize(416, 416),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(416, 416),
            ToTensorV2()
        ])
    

# =============================================================================
# DataLoader Creator Function
# =============================================================================
def get_dataloader(csv_fname, images_dir, batch_size=32, train=True):
    alb_transform = get_albumentations_transform(train=train)

    dataset = SixrayWrapper(csv_fname, images_dir, alb_transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)