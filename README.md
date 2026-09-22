# Cross-Domain Affective Transfer Learning: Dog Emotion Classification

This repository investigates cross-domain transfer learning efficacy on canine affective states using deep convolutional neural networks. I evaluate whether feature representations pretrained on human facial affect (**AffectNet**) generalize better to animal expressions than representations learned on generic natural objects (**ImageNet**).

The project implements automated Bayesian hyperparameter tuning via **Optuna** to optimize learning rate, regularization, layer freezing depth, and optimization algorithms across both paradigms.

---

## Experimental Setup

* **Architecture:** ResNet-50 backbone with custom classification head (Dropout + Fully Connected layer).
* **Dataset:** Dog Emotion Dataset via Kaggle (classes: *angry*, *happy*, *sad*, *relaxed*).
* **Hyperparameter Optimization:** Optuna Bayesian search over:
  * Learning rate: `[1e-5, 1e-3]` (log scale)
  * Weight decay: `[1e-6, 1e-3]` (log scale)
  * Optimizers: Adam, AdamW, SGD
  * Batch sizes: 8, 16, 32
  * Layer freeze depth: Blocks 0 to 3
  * Dropout rate: `[0.0, 0.5]`
* **Training Dynamics:** Cross-Entropy Loss, `ReduceLROnPlateau` scheduling (patience=3, factor=0.5), and early stopping (patience=7).

---

## Results Summary

| Backbone Pretraining | Best Optimizer | Freeze Depth | Dropout | Best Val Acc (%) | Convergence Point | Early Stopping |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ImageNet (Default)** | AdamW | 3 blocks (`conv2_x`, `conv3_x`, `conv4_x`) | ~0.0208 | **88.38%** | Epoch 11 | Epoch 18 |
| **AffectNet (Human Affect)** | AdamW | 1 block (`conv2_x`) | ~0.1393 | **74.38%** | Epoch 21 | Epoch 28 |


### Key Observations
* **ImageNet Baseline:** Converged rapidly at Epoch 11 with training loss declining from 0.8953 down to 0.0043, showing that broad object features establish a strong baseline for facial geometry.
* **AffectNet Transfer:** Reached peak validation accuracy of 74.38% at Epoch 21 while validation loss climbed to 1.3456. This highlights surrogate loss behavior where increasing cross-entropy penalty from confident outlier errors does not prevent the model from improving top-1 discrete classification accuracy.
* **Backbone Plasticity:** ImageNet converged best with 3 frozen residual blocks, whereas AffectNet required high structural plasticity (only 1 frozen block and a higher dropout rate) to map human emotion features onto canine anatomy.

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
