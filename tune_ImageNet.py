import os

import kagglehub
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ------------------------------------------------------
# 1. DOWNLOAD KAGGLE DATASET
# ------------------------------------------------------
print("Downloading Dog Emotion dataset...")
dataset_path = kagglehub.dataset_download("danielshanbalico/dog-emotion")
print("Dataset downloaded to:", dataset_path)

# Inside this folder is where angry/, happy/, sad/, relaxed/ exist.
data_dir = os.path.join(dataset_path, "Dog Emotion")


# ------------------------------------------------------
# 2. DEVICE
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------------------
# 3. TRANSFORMS
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
# 4. LOAD DATASET
# ------------------------------------------------------
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(full_dataset.classes)

print("Detected classes:", full_dataset.classes)
print("Total images:", len(full_dataset))


# ------------------------------------------------------
# 5. OPTUNA OBJECTIVE
# ------------------------------------------------------
def objective(trial):

    # Hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    freeze_layers = trial.suggest_categorical("freeze_layers", [0, 1, 2, 3])
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)

    # -----------------------------
    # Dataset split (80/20)
    # -----------------------------
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Load ResNet50
    # -----------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze early layers
    children = list(model.children())
    for child in children[:freeze_layers]:
        for param in child.parameters():
            param.requires_grad = False

    # Replace FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer selection
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # -----------------------------
    # Training loop (short)
    # -----------------------------
    EPOCHS = 20

    for epoch in range(EPOCHS):

        # Training
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


# ------------------------------------------------------
# 6. RUN OPTUNA
# ------------------------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)   # increase later on SHARCNET

print("\nBest hyperparameters:")
print(study.best_params)
print("Best accuracy:", study.best_value)
