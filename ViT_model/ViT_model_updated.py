import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import timm

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,          
)
import matplotlib.pyplot as plt

class XRayWeaponDataset(Dataset):
    """
    Dataset for X-ray luggage images with multi-label targets corresponding
    to weapon classes: knife, gun, wrench, pliers, scissors.
    """
    def __init__(self, annotations_dir, images_dir, transform=None, classes=None):
        self.annotations = glob.glob(os.path.join(annotations_dir, "*.xml"))
        self.images_dir = images_dir
        self.transform = transform
        self.classes = classes or ["knife", "gun", "wrench", "pliers", "scissors"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        xml_path = self.annotations[idx]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.find("filename").text
        img_path = os.path.join(self.images_dir, filename)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Multi-hot label vector
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            if name in self.class_to_idx:
                label[self.class_to_idx[name]] = 1.0
        return img, label

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)                              
            preds = (torch.sigmoid(outputs) > threshold)        
            correct += (preds == labels.bool()).sum().item()
            total += labels.numel()
    return correct / total

def train(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {epoch_loss:.4f}  Val Acc: {val_acc:.4f}")

def main():
    # Paths to your data
    # After unpacking the SIXRay dataset, you would need to set the directories in a similar way to this
    # Classes must remain unchanged, unless one wants to experiment with more objects. In that case, and XML file with ground
    # truth annotations, negative/positive images will be required
    annotations_dir = r"C:\Users\rakin\OneDrive - Colgate University\Documents\Final Project\pretrainedproject\positive-Annotation"
    images_dir      = r"C:\Users\rakin\OneDrive - Colgate University\Documents\Final Project\pretrainedproject\JPEGImage"
    classes         = ["knife", "gun", "wrench", "pliers", "scissors"]

    # Hyperparameters
    # Feel free to change the parameters here as you wish
    batch_size = 16
    val_split  = 0.2
    epochs     = 5 
    lr         = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Dataset + split
    dataset   = XRayWeaponDataset(annotations_dir, images_dir, transform, classes)
    val_size  = int(len(dataset) * val_split)
    train_size= len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    # Load ViT with new head
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(classes))
    model.to(device)

    # Baseline
    baseline_acc = evaluate(model, val_loader, device)
    print(f"Pre-fine-tune accuracy: {baseline_acc:.4f}")

    # Fine-tuning
    train(model, train_loader, val_loader, device, epochs, lr)

    # Final accuracy
    finetuned_acc = evaluate(model, val_loader, device)
    print(f"Post-fine-tune accuracy: {finetuned_acc:.4f}\n")

    # === Metrics, Confusion Matrix & F1 Scores ===
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs    = imgs.to(device)
            outputs = model(imgs)
            preds   = (torch.sigmoid(outputs) > 0.5).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    y_true = torch.vstack(all_labels).numpy()
    y_pred = torch.vstack(all_preds).numpy()

    # Flatten for overall binary confusion
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Compute standard metrics
    cm      = confusion_matrix(y_true_flat, y_pred_flat)
    acc     = accuracy_score(y_true_flat, y_pred_flat)
    report  = classification_report(y_true, y_pred, target_names=classes, zero_division=0)

    # Compute F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None)
    f1_micro     = f1_score(y_true, y_pred, average='micro')
    f1_macro     = f1_score(y_true, y_pred, average='macro')
    f1_weighted  = f1_score(y_true, y_pred, average='weighted')

    # Print out
    print(f"Overall (flattened) accuracy: {acc:.4f}\n")
    print("Per-class precision, recall & F1:\n")
    print(report)
    print("=== F1 Scores ===")
    for cls, f1 in zip(classes, f1_per_class):
        print(f"{cls:>8} F1: {f1:.4f}")
    print(f"\nMicro F1:    {f1_micro:.4f}")
    print(f"Macro F1:    {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}\n")

    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (flattened)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    tick_labels = ["No Weapon","Weapon"]
    plt.xticks([0,1], tick_labels)
    plt.yticks([0,1], tick_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

