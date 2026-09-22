# Cross-Domain Affective Transfer Learning: Dog Emotion Classification

This repository investigates cross-domain transfer learning efficacy on canine affective states using deep convolutional neural networks[cite: 16]. We evaluate whether feature representations pretrained on human facial affect (**AffectNet**) generalize better to animal expressions than representations learned on generic natural objects (**ImageNet**)[cite: 16].

The project implements automated Bayesian hyperparameter tuning via **Optuna** to optimize learning rate, regularization, layer freezing depth, and optimization algorithms across both paradigms[cite: 16].

All pipeline engineering, layer surgery, Optuna tuning, and training scripts in this repository were independently developed[cite: 11, 12, 13, 14, 15].

---

## Experimental Setup

* **Architecture:** ResNet-50 backbone with custom classification head (Dropout + Fully Connected layer).
* **Dataset:** Dog Emotion Dataset via Kaggle (classes: *angry*, *happy*, *sad*, *relaxed*)[cite: 11, 12, 13, 14, 16].
* **Hyperparameter Optimization:** Optuna Bayesian search over:
  * Learning rate: `[1e-5, 1e-3]` (log scale)
  * Weight decay: `[1e-6, 1e-3]` (log scale)[cite: 13, 14]
  * Optimizers: Adam, AdamW, SGD[cite: 13, 14]
  * Batch sizes: 8, 16, 32[cite: 13, 14]
  * Layer freeze depth: Blocks 0 to 3[cite: 13, 14, 16]
  * Dropout rate: `[0.0, 0.5]`[cite: 13, 14]
* **Training Dynamics:** Cross-Entropy Loss, `ReduceLROnPlateau` scheduling (patience=3, factor=0.5), and early stopping (patience=7)[cite: 11, 12, 16].

---

## Results Summary

| Backbone Pretraining | Best Optimizer | Freeze Depth | Dropout | Best Val Acc (%) | Convergence Point | Early Stopping |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ImageNet (Default)** | AdamW | 3 blocks (`conv2_x`, `conv3_x`, `conv4_x`) | ~0.0208 | **88.38%** | Epoch 11 | Epoch 18 |
| **AffectNet (Human Affect)** | AdamW | 1 block (`conv2_x`) | ~0.1393 | **74.38%** | Epoch 21 | Epoch 28 |

[cite: 16]

### Key Observations
* **ImageNet Baseline:** Converged rapidly at Epoch 11 with training loss declining from 0.8953 down to 0.0043, showing that broad object features establish a strong baseline for facial geometry[cite: 16].
* **AffectNet Transfer:** Reached peak validation accuracy of 74.38% at Epoch 21 while validation loss climbed to 1.3456[cite: 16]. This highlights surrogate loss behavior where increasing cross-entropy penalty from confident outlier errors does not prevent the model from improving top-1 discrete classification accuracy[cite: 16].
* **Backbone Plasticity:** ImageNet converged best with 3 frozen residual blocks, whereas AffectNet required high structural plasticity (only 1 frozen block and a higher dropout rate) to map human emotion features onto canine anatomy[cite: 16].

---

## Repository Structure

```text
├── train_ImageNet.py          # Fine-tuning pipeline using ImageNet weights
├── train_AffectNet.py         # Fine-tuning pipeline using AffectNet weights
├── tune_ImageNet.py           # Optuna study for ImageNet backbone
├── tune_AffectNet.py          # Optuna study for AffectNet backbone
├── inspect_weights.ipynb      # State-dict inspection and layer verification
├── requirements.txt           # Package dependencies
└── README.md
