# ShiftGuard-10

Classification pipeline for 32×32 low-resolution images across 10 classes, built for the ShiftGuard-10 competition (EE708). The core objective is maintaining a high Macro F1-score when test data exhibits both class imbalance and distribution shifts.

## Problem and Constraints

- **Input:** 32×32 RGB images
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Target Metric:** Macro F1-score
- **Main Hurdles:** Low spatial resolution, heavy class skew in training batches, and domain shift between the training set and hidden evaluation sets.

## What Worked

Rather than using an overly deep network that easily overfits on 32×32 inputs, the pipeline centers on **WideResNet-28-10** combined with regularization and sampling strategies:

1. **Handling Skewed Classes:**
   - Balanced mini-batches using PyTorch `WeightedRandomSampler` based on reciprocal class frequencies.
   - Frequency-adjusted cross-entropy loss to prevent majority classes from dominating gradient updates.

2. **Data Augmentation:**
   - Standard spatial transforms: random crops (padding=4) and horizontal flips.
   - AutoAugment (CIFAR-10 policy) and random cutout/patch masking to force the network to look beyond local texture artifacts.

3. **Regularization:**
   - **MixUp & CutMix:** Blending input pairs and target distributions during training to soften decision boundaries.
   - **Label Smoothing (0.1):** Prevents overconfident predictions on noisy or ambiguous samples.

4. **Optimization & Ensembling:**
   - Optimizer: AdamW with cosine learning rate decay and a short warm-up phase.
   - **Stochastic Weight Averaging (SWA):** Takes a running average of checkpoints during the later training epochs for a flatter loss surface.
   - **Test-Time Augmentation (TTA):** Averages predictions over original and flipped test images to trim prediction variance.

## Repository Structure

```text
ShiftGuard-10/
├── notebooks/
│   └── Shiftguard-10.ipynb   # Complete training, validation, and inference code
├── reports/
│   └── ShiftGuard-10.pdf     # Project report with ablation results and analysis
├── results/
│   └── submission.csv        # Output predictions formatted for evaluation
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Expects the challenge dataset extracted into:
```text
shift-guard-10-robust-image-classification-challenge/
├── train/
├── test/
└── train_labels.csv
```

### 3. Running Training and Inference

Open and execute the notebook:

```bash
jupyter notebook notebooks/Shiftguard-10.ipynb
```

The notebook seeds all RNGs, sets up data loaders, trains the WideResNet model with SWA, runs TTA on the test set, and writes predictions directly to `results/submission.csv`.
