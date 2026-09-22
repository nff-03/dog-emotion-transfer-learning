# Cross-Species Emotion Recognition in Animals via Domain-Adaptive Transfer Learning

This repository contains the complete PyTorch implementation, training pipelines, and automated hyperparameter optimization experiments for investigating cross-species affective computing[cite: 11, 12, 13, 14, 16]. Specifically, this project evaluates domain-adaptive transfer learning by comparing feature transferability from human facial affect models (**AffectNet**) against standard generic visual representations (**ImageNet**) on canine facial expressions.

All data pipelines, layer surgery, Optuna Bayesian optimization routines, and training scripts in this repository were independently designed and developed[cite: 11, 12, 13, 14, 15].

---

## Key Experimental Findings

* **ImageNet Initialization (Non-Domain Adaptive Baseline):** Reached a peak validation accuracy of **88.38%**, converging at Epoch 11. Training loss decreased from 0.8953 to 0.0043 across 18 epochs before early stopping triggered at Epoch 18[cite: 16].
* **AffectNet Initialization (Domain-Adaptive Transfer):** Achieved a validation accuracy of **74.38%**, converging at Epoch 21 (early stopping triggered at Epoch 28)[cite: 16]. 
* **Loss-Accuracy Divergence Phenomenon:** While AffectNet training loss steadily declined to 0.0286, validation loss escalated to 1.3456 past Epoch 15 while validation accuracy continued rising to its peak at Epoch 21[cite: 16]. This demonstrates surrogate loss behavior where penalization on hard/outlier misclassifications increases without degrading top-1 discrete classification performance[cite: 16].
* **Structural Plasticity Requirements:** ImageNet weights converged optimally with a rigid backbone (3 residual blocks frozen)[cite: 16], whereas AffectNet required higher structural plasticity (only 1 residual block frozen and a 6.7× higher dropout rate) to project human expression geometry onto canine facial anatomy[cite: 16].

---

## Performance & Convergence Summary

| Initialization Source | Best Validation Accuracy | Convergence Epoch | Early Stopping Epoch | Frozen Blocks | Best Optimizer | Final Learning Rate | Weight Decay | Dropout |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ImageNet Baseline** | **88.38%** | Epoch 11 | Epoch 18 | 3 (`conv2_x`, `conv3_x`, `conv4_x`) | AdamW | $9.3596 \times 10^{-5}$ | $8.94 \times 10^{-4}$ | 0.0208 |
| **AffectNet Transfer** | **74.38%** | Epoch 21 | Epoch 28 | 1 (`conv2_x`) | AdamW | $2.8047 \times 10^{-4}$ | $9.92 \times 10^{-4}$ | 0.1393 |

[cite: 16]

---

## Dataset & Preprocessing

* **Dataset:** Kaggle Dog Emotion Dataset (`danielshanbalico/dog-emotion`) containing 4 target emotion classes: *happy*, *sad*, *angry*, and *relaxed*[cite: 16].
* **Input Resolution:** Resized to $224 \times 224$ pixels[cite: 16].
* **Normalization:** Standard ImageNet channel-wise normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)[cite: 16].
* **Splits & Augmentations:** 80% train / 20% validation split, utilizing random horizontal flipping, rotation, and color jitter[cite: 16].

---

## Architecture & Engineering Implementation

The architecture replaces the standard 1000-class fully connected head of ResNet-50 with a dropout layer and a 4-unit linear projection[cite: 16]. Layer surgery was implemented to map non-standard AffectNet checkpoints into torchvision-compatible state dict keys.

Hyperparameter configurations were obtained via 50 Optuna trials evaluating[cite: 16]:
* **Optimizers:** Adam, AdamW, SGD[cite: 13, 14, 16]
* **Learning Rates:** Log-uniform continuous search $\left[10^{-5}, 10^{-3}\right]$
* **Weight Decay:** Log-uniform continuous search $\left[10^{-6}, 10^{-3}\right]$[cite: 13, 14]
* **Batch Sizes:** Categorical selection $\in \{8, 16, 32\}$[cite: 13, 14]
* **Freeze Depths:** Discrete layer freezing $\in \{0, 1, 2, 3\}$ residual blocks[cite: 13, 14, 16]
* **Regularization:** Fully connected dropout rate $\in [0.0, 0.5]$[cite: 13, 14]

The training loop implements Cross-Entropy Loss, `ReduceLROnPlateau` dynamic scheduling (factor=0.5, patience=3 epochs), and early stopping (patience=7 epochs).

---

## Repository Structure

```text
├── train_ImageNet.py          # Training pipeline for ImageNet baseline
├── train_AffectNet.py         # Training pipeline for AffectNet transfer model
├── tune_ImageNet.py           # Optuna Bayesian search for ImageNet weights
├── tune_AffectNet.py          # Optuna Bayesian search for AffectNet weights
├── inspect_weights.ipynb      # State-dict inspection and layer verification
├── requirements.txt           # Python package dependencies
└── README.md
