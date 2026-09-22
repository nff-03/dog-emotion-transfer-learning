import os

import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ------------------------------------------------------
# 1. Device
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------
# 2. Dataset path (update if needed)
# ------------------------------------------------------
print("Downloading Dog Emotion dataset...")
dataset_path = kagglehub.dataset_download("danielshanbalico/dog-emotion")
print("Dataset downloaded to:", dataset_path)

data_dir = os.path.join(dataset_path, "Dog Emotion")

# ------------------------------------------------------
# 3. Transforms for ResNet-50
# ------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ------------------------------------------------------
# 4. Load full dataset and split
# ------------------------------------------------------
dataset = datasets.ImageFolder(data_dir, transform=transform)

num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

# 80/20 split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ------------------------------------------------------
# 5. Hyperparameters from Optuna
# ------------------------------------------------------
best_hparams = {
    'optimizer': 'AdamW',
    'lr': 9.359683776188821e-05,
    'weight_decay': 0.0008943160562530289,
    'batch_size': 16,
    'freeze_layers': 3,
    'dropout': 0.020864564204807506
}

batch_size = best_hparams["batch_size"]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# ------------------------------------------------------
# 6. Build ResNet-50 with dropout + layer freezing
# ------------------------------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze first N ResNet blocks
freeze_depth = best_hparams["freeze_layers"]
if freeze_depth >= 1: 
    for p in model.layer1.parameters(): p.requires_grad = False
if freeze_depth >= 2: 
    for p in model.layer2.parameters(): p.requires_grad = False
if freeze_depth >= 3: 
    for p in model.layer3.parameters(): p.requires_grad = False
if freeze_depth >= 4: 
    for p in model.layer4.parameters(): p.requires_grad = False

# Add dropout before final FC
dropout_rate = best_hparams["dropout"]
model.fc = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

# ------------------------------------------------------
# 7. Loss & Optimizer (AdamW)
# ------------------------------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=best_hparams["lr"],
    weight_decay=best_hparams["weight_decay"]
)

# ------------------------------------------------------
# 8. Training Loop
# ------------------------------------------------------
EPOCHS = 50
patience = 7
no_improve = 0

best_val_loss = float("inf")
best_val_acc = 0.0

# Scheduler: reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)


for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---------- TRAIN ----------
    model.train()
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / len(val_dataset)

    print(f"Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%")

    # Step LR scheduler
    scheduler.step(val_loss)

    # ---------- CHECK FOR IMPROVEMENT ----------
    improved = False

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        improved = True

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        improved = True

    if improved:
        print("✨ Improvement detected — saving model...")
        torch.save(model.state_dict(), "best_resnet50_dog_emotion.pth")
        no_improve = 0
    else:
        no_improve += 1
        print(f"No improvement for {no_improve}/{patience} epochs")

    # ---------- EARLY STOPPING ----------
    if no_improve >= patience:
        print("Early stopping triggered!")
        break

# ------------------------------------------------------
# 9. Result Summary
# ------------------------------------------------------
print("\nTraining complete!")
print(f"Best Validation Accuracy: {100*best_val_acc:.2f}%")
print(f"Best Validation Loss: {best_val_loss:.4f}")