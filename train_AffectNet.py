import os

import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ------------------------------------------------------
# 1. BEST HYPERPARAMETERS
# ------------------------------------------------------
best_hparams = {
    'optimizer': 'AdamW',
    'lr': 0.00028047318140045927,
    'weight_decay': 0.000992483903487377,
    'batch_size': 16,
    'freeze_layers': 1,
    'dropout': 0.13937721321804394
}

# ------------------------------------------------------
# 2. DEVICE
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------
# 3. DOWNLOAD DATASET
# ------------------------------------------------------
print("Downloading Dog Emotion dataset...")
dataset_path = kagglehub.dataset_download("danielshanbalico/dog-emotion")
print("Dataset downloaded to:", dataset_path)

data_dir = os.path.join(dataset_path, "Dog Emotion")

# ------------------------------------------------------
# 4. TRANSFORMS
# ------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ------------------------------------------------------
# 5. LOAD DATASET
# ------------------------------------------------------
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(full_dataset.classes)

print("Detected classes:", full_dataset.classes)

# 80/20 split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=best_hparams['batch_size'],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=best_hparams['batch_size'],
    shuffle=False
)

# ------------------------------------------------------
# 6. LOAD AFFECTNET PRETRAINED RESNET50
# ------------------------------------------------------
def load_affectnet_resnet(num_classes):
    model = models.resnet50(weights=None)

    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load your local pretrained weights
    # Make sure this file exists in the current directory!
    state_dict = torch.load(
        "FER_static_ResNet50_AffectNet.pt",
        map_location="cpu"
    )

    # Rename keys to match torchvision
    rename_map = {
        "conv_layer_s2_same.weight": "conv1.weight",
        "batch_norm1.weight": "bn1.weight",
        "batch_norm1.bias": "bn1.bias",
        "batch_norm1.running_mean": "bn1.running_mean",
        "batch_norm1.running_var": "bn1.running_var",
    }

    for old, new in rename_map.items():
        if old in state_dict:
            state_dict[new] = state_dict.pop(old)

    model.load_state_dict(state_dict, strict=False)
    return model

# ------------------------------------------------------
# 7. MODEL SETUP
# ------------------------------------------------------
model = load_affectnet_resnet(num_classes)

# Freeze early layers
children = list(model.children())
for child in children[:best_hparams['freeze_layers']]:
    for param in child.parameters():
        param.requires_grad = False

# Add dropout to FC
model.fc = nn.Sequential(
    nn.Dropout(best_hparams['dropout']),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

# ------------------------------------------------------
# 8. LOSS, OPTIMIZER, SCHEDULER
# ------------------------------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=best_hparams['lr'],
    weight_decay=best_hparams['weight_decay']
)

# Scheduler: reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

# ------------------------------------------------------
# 9. TRAINING LOOP
# ------------------------------------------------------
EPOCHS = 50
patience = 7
patience_counter = 0

best_val_acc = 0.0
best_val_loss = float("inf")  # Added to track loss improvement

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total

    # Step scheduler based on loss
    scheduler.step(val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Patience: {patience_counter}/{patience}"
    )

    # --- CHECK FOR IMPROVEMENT (Acc OR Loss) ---
    improved = False

    # Check if loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        improved = True
    
    # Check if accuracy improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        improved = True

    if improved:
        print("✨ Improvement detected (Loss or Acc) — saving model...")
        torch.save(model.state_dict(), "best_affectnet_resnet50.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter}/{patience} epochs")

    # Early stopping trigger
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

print("Training complete")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Best Validation Loss: {best_val_loss:.4f}")
