import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def custom_collate_fn(batch):
    """
    Collates the batch such that images are stacked and targets
    are returned as a list.
    
    Args:
        batch: list of tuples (image, targets)
    Returns:
        images: a stacked tensor of shape [B, 3, 416, 416]
        targets: a list of tensors (each can have different shape, e.g. [N, 4])
    """
    images = torch.stack([item[0] for item in batch], 0)
    targets = [item[1] for item in batch]
    return images, targets


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
    def __init__(self, csv_fname, images_dir, annotations_dir, transform=None):
        self.fnames = pd.read_csv(csv_fname, delimiter=',', header=0)
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        # This mapping converts XML class names to numeric classes.
        self.class_id = {"Gun": 0, "Knife": 1, "Wrench": 2, "Pliers": 3, "Scissors": 4, "Hammer": 5}

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # Get the image file name from CSV (assume the CSV has a column "name")
        row = self.fnames.iloc[idx]
        file_name = row['name']  # e.g., "P00007"
        
        image_path = os.path.join(self.images_dir, file_name + ".jpg")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Initialize an empty list for bounding boxes.
        bounding_boxes = []
        # Determine if image is positive by its name (if it starts with 'P', we assume positive)
        if file_name[0] == 'P':
            xml_path = os.path.join(self.annotations_dir, file_name + ".xml")
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    # Get original image size from XML we use these for normalization.
                    size_node = root.find('size')
                    orig_width = float(size_node.find('width').text)
                    orig_height = float(size_node.find('height').text)
                    
                    for obj in root.findall('object'):
                        name_node = obj.find('name')
                        if name_node is None:
                            continue
                        class_name = name_node.text
                        if class_name not in self.class_id:
                            continue
                        c = self.class_id[class_name]
                        bndbox = obj.find('bndbox')
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)
                        # Convert to YOLO format (normalized center x, center y, width, height)
                        x_center = ((xmin + xmax) / 2.0) / orig_width
                        y_center = ((ymin + ymax) / 2.0) / orig_height
                        box_w = (xmax - xmin) / orig_width
                        box_h = (ymax - ymin) / orig_height
                        bounding_boxes.append([c, x_center, y_center, box_w, box_h])
                except Exception as e:
                    print(f"Error parsing XML for {file_name}: {e}")
        # For negative images, no bounding boxes needed.
        bounding_boxes = np.array(bounding_boxes, dtype=np.float32)

        sample = {"image": image_np, "bounding_boxes": bounding_boxes}
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
    boxes = sample["bounding_boxes"]
    if boxes.size == 0:
        bboxes = []
        labels = []
    else:
        bboxes = boxes[:, 1:].tolist()  # Extract box coordinates.
        labels = boxes[:, 0].tolist()     # Extract class labels.
    transformed = alb_transform(image=image, bboxes=bboxes, labels=labels)
    # Ensure the transformed image is a float tensor due to prior error.
    transformed["image"] = transformed["image"].float()
    return {"image": transformed["image"], "bounding_boxes": transformed["bboxes"], "labels": transformed["labels"]}
    
# =============================================================================
# A Wrapper Dataset that applies our custom_transform to the raw SixrayDataset output.
# =============================================================================
class SixrayWrapper(Dataset):
    def __init__(self, csv_fname, images_dir, annotations_dir, alb_transform):
        self.dataset = SixrayDataset(csv_fname, images_dir, annotations_dir, transform=None)
        self.alb_transform = alb_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]  # returns {"image": ..., "bounding_boxes": ..., "labels": ...}
        sample = custom_transform(sample, self.alb_transform)
        # Now sample["bounding_boxes"] is a list of bboxes [x_center, y_center, w, h]
        # and sample["labels"] is a list of class ids.
        if len(sample["bounding_boxes"]) > 0:
            # Convert both lists to tensors and concatenate along dimension 1.
            labels = torch.tensor(sample["labels"]).unsqueeze(1).float()  # shape: [N, 1]
            bboxes = torch.tensor(sample["bounding_boxes"]).float()       # shape: [N, 4]
            targets = torch.cat([labels, bboxes], dim=1)  # shape: [N, 5]
        else:
            targets = torch.empty((0, 5))
        return sample["image"], targets

    
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
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(416, 416),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))
    

# =============================================================================
# DataLoader Creator Function
# =============================================================================
def get_dataloader(csv_fname, images_dir, annotations_dir, batch_size=32, train=True):
    alb_transform = get_albumentations_transform(train=train)
    dataset = SixrayWrapper(csv_fname, images_dir, annotations_dir, alb_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)