# Can a GAN Discriminator Detect AI-Generated Images?

Train a TensorFlow GAN and test whether its discriminator transfers from its own generated images to Stable Diffusion image detection.

---

## Overview

* **Task:** Binary classification of real versus AI-generated images.
* **Dataset:** 18,000 CIFAKE images; 32 × 32 RGB pixels (3,072 pixel-channel values per image).
* **Selected Model:** **CNN from scratch** (test **ROC-AUC = 0.9433**), selected by validation ROC-AUC.
* **Baseline Comparison:** Always-real baseline achieved ROC-AUC = 0.5000 and accuracy = 50.00% on the balanced test set.

**Research question:** Does adversarial pretraining improve a detector of images from a different generation method?

The raw discriminator achieved **0.999997 ROC-AUC on fresh outputs from its own GAN**, but only **0.5890 on Stable Diffusion images**. GAN initialization did not improve the supervised detector's primary test ROC-AUC in this fixed-budget, one-seed experiment. Recognizing one weak generator's artifacts is insufficient evidence of general AI-image detection.

---

## Key Results

### Training Validation

One fixed validation partition was used; cross-validation and fold SD are not applicable. The raw GAN uses its fixed final epoch. Supervised checkpoints use the highest validation ROC-AUC, with earlier epochs winning ties.

| Model | Validation ROC-AUC | Selected epoch |
| :--- | ---: | ---: |
| Raw GAN discriminator | 0.6000 | 10 |
| CNN from scratch | 0.9397 | 6 |
| GAN-initialized detector | 0.9362 | 6 |
| Always-real baseline | 0.5000 | N/A |

### Final Holdout Evaluation (4,000 Images)

The primary test contains 2,000 real and 2,000 Stable Diffusion images. Positive class = AI-generated; the decision threshold was fixed at 0.5 before evaluation.

| Model | ROC-AUC | Macro-F1 | Accuracy |
| :--- | ---: | ---: | ---: |
| Raw GAN discriminator | 0.5890 | 0.3564 | 50.58% |
| CNN from scratch | 0.9433 | 0.8710 | 87.10% |
| GAN-initialized detector | 0.9381 | 0.8469 | 84.78% |
| Always-real baseline | 0.5000 | 0.3333 | 50.00% |

The selected detector's ROC-AUC has a conditional 95% bootstrap interval of **[0.9361, 0.9494]**. The paired ROC-AUC difference, GAN-initialized minus scratch, is **-0.0052**, with interval **[-0.0088, -0.0018]**. These are 500 class-stratified resamples of the same held-out predictions. They do not measure variability across training seeds, datasets, or generators, and assume the sampled images are sufficiently independent.

### Generator Transfer and Compression Stress Test

| Model | CIFAKE ROC-AUC | JPEG Q50 ROC-AUC | Own GAN ROC-AUC |
| :--- | ---: | ---: | ---: |
| Raw GAN discriminator | 0.5890 | 0.5858 | 1.0000 |
| CNN from scratch | 0.9433 | 0.9323 | 0.9174 |
| GAN-initialized detector | 0.9381 | 0.9193 | 0.9647 |
| Always-real baseline | 0.5000 | 0.5000 | 0.5000 |

JPEG recompression changes the selected detector's accuracy from **87.10% to 83.73%**, and its false-positive rate from **13.05% to 24.80%**. The same real test images are reused across conditions; these are related evaluations, not independent replications.

### Experimental Design

1. **Adversarial pretraining:** A generator maps 64-dimensional Gaussian noise to 32 × 32 RGB images. A convolutional discriminator learns real = 0 and generated = 1. Both train jointly for 10 epochs on the 6,000 real training images only.
2. **Raw discriminator:** Freeze the GAN's final discriminator. No Stable Diffusion images are used to train this model.
3. **Scratch detector:** Train the same discriminator architecture from random initialization on 6,000 real and 6,000 Stable Diffusion training images for six epochs.
4. **GAN-initialized detector:** Initialize that architecture with the GAN discriminator's learned weights, then fine-tune all weights on the same labeled images for the same six supervised epochs.
5. **Evaluation:** Select supervised checkpoints and the reported model using 2,000 validation images before scoring the test set. Also evaluate JPEG quality 50 and 2,000 fresh GAN outputs against the same 2,000 real test images. GAN test images are quantized and JPEG-encoded at quality 95 before evaluation.

All three discriminators contain 167,649 parameters: three stride-2 convolutional layers (32/64/128 channels), leaky ReLU, dropout 0.25, and a dense output. The generator has 305,027 parameters and uses transpose convolutions, batch normalization, ReLU, and tanh. Input pixels are decoded to RGB and scaled to [-1, 1]. Batch size is 128; seed is 42. GAN Adam uses learning rate 0.0002 and beta1 = 0.5; supervised Adam uses 0.001. Gradient global norm is clipped at 1.0.

The two supervised models share architecture, data, and supervised epoch budget. The GAN-initialized model also receives adversarial pretraining, so total compute is not matched. No pretrained external weights or paid model APIs are used. The GAN receives a short fixed training budget; its displayed outputs are highly repetitive checkerboard patterns, not realistic photographs. Its near-perfect own-generator detection is therefore an easy artifact-detection result. GAN losses alone do not establish visual quality, and this unsuccessful run does not establish that GAN pretraining generally cannot help.

---

## Data Summary

* **Source:** [CIFAKE by Jordan J. Bird and Ahmad Lotfi](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images), pinned commit `e112a942abaecd02b6b1f6f646c807d56be8fb62`. Real images originate from CIFAR-10; synthetic images were generated by the dataset authors using Stable Diffusion 1.4.
* **Data Quality:** 18,000 readable RGB images; 24 exact-duplicate candidates skipped; 18,000 unique decoded-pixel hashes. No pixel imputation. Near duplicates and shared content are not ruled out.
* **Partitions:** 12,000 training, 2,000 validation, and 4,000 test images, balanced by real/AI label. Training and validation come only from the source training split; test images come only from its test split.
* **Selection:** Within each original split/class, sort SHA-256 of `42:path` and retain the first images with unique decoded-pixel hashes. The first 6,000 retained training images per class form training and the next 1,000 form validation. Paths and both file/pixel hashes are recorded in `sample_manifest.csv.gz`.
* **Integrity:** All 18,000 decompressed manifest records were checked against the pinned author archive. Original images and the source archive are referenced, not republished as a dataset snapshot.
* **Interpretability:** Error examples show failure cases. They are not causal explanations or evidence of which visual features the model uses; tabular permutation importance and feature correlations are not reported.

### Error Analysis and Limits

At threshold 0.5, the selected detector falsely flags **261 real images** and misses **255 synthetic images**. Every selected-model error is recorded in `error_analysis.csv`; all 42,000 model scores across validation and test conditions are in `predictions.csv`.

The classifier can exploit generator, resizing, or compression artifacts. It has only been tested on a small 32 × 32 benchmark, this one locally trained GAN, and one diffusion-model source. Its scores are not calibrated probabilities of authenticity. High-resolution photographs, screenshots, newer generators, edited images, and different class balances need separate evaluation. This study does not establish a reliable detector for arbitrary online pictures.

### Artifact Verification

Saved classifier checkpoints are checked against all recorded validation and test-condition scores. Regenerating the 2,000 GAN test images produced 1,989 byte-identical images and small pixel differences in 11 images (maximum RGB-channel difference 5/255; mean absolute difference 0.000256 on the 0–255 scale). The original experimental outputs are preserved in `generated_test_images.npz` for exact own-GAN evaluation. This archive contains generated experimental images, not source CIFAKE images. Its complete decompressed array is checked against the original run. All 18,000 compressed source-manifest records are separately checked against the public author archive.

---

## Visualizations

| Dataset examples | GAN training progression |
| :---: | :---: |
| ![Dataset examples](figures/dataset-examples.png) | ![GAN progression](figures/gan-progress.png) |

| Training curves | Primary model comparison |
| :---: | :---: |
| ![Learning curves](figures/learning-curves.png) | ![Model comparison](figures/model-comparison.png) |

| ROC curves | Confusion matrices |
| :---: | :---: |
| ![ROC curves](figures/roc-curve.png) | ![Confusion matrices](figures/confusion-matrix.png) |

| Transfer and compression sensitivity | Selected detector errors |
| :---: | :---: |
| ![Robustness](figures/robustness.png) | ![Error examples](figures/error-examples.png) |

---

## Repository Structure

```text
├── figures/                    # Eight actual diagnostic and comparison plots
├── analysis.ipynb              # Executed analysis of saved predictions
├── prepare_data.py             # Pinned download, selection, and integrity checks
├── train.py                    # GAN, detector training, and fixed evaluations
├── verify.py                   # Full checkpoint and source-manifest verification
├── build_report.py             # Figures, uncertainty analysis, and report
├── predict.py                  # Score an image using a saved detector
├── requirements.txt            # Pinned direct dependencies
├── audit.json                  # Actual training environment, dates, and hashes
├── verification_environment.txt # Environment used to reload and verify models
├── data_sources.json           # Archive identity, counts, and selection policy
├── sample_manifest.csv.gz      # Every selected source path and integrity hash
├── duplicates_removed.json     # Skipped duplicate candidates
├── study_config.json           # Frozen training/evaluation choices
├── training_history.csv        # Actual GAN losses and detector validation scores
├── selection.json              # Validation-only checkpoint/model choices
├── predictions.csv             # All validation and test-condition scores
├── generated_test_images.npz    # Original 2,000 synthetic test outputs
├── error_analysis.csv          # Selected detector's 516 primary-test errors
├── metrics.json                # Full validation and holdout metrics
├── uncertainty.json            # Conditional bootstrap intervals
├── *.weights.npz               # Generator and all three detector checkpoints
├── verification.json           # Full prediction and source-check results
├── DATA_LICENSE.txt            # Dataset authors' license statement
├── SOURCE.md                   # Dataset and research references
├── LEARNING_NOTES.md           # Unfilled human review prompts
├── repository_metadata.json    # Suggested standalone name, About, and topics
├── report.html                 # Report with relative links to figure files
├── project.json                # Completed-study manifest and artifact hashes
└── README.md
```

Training ran on 28 September 2026; documentation and publication followed on 29 September. Automated implementation, execution, and analysis by OpenAI Codex for Saud Alotaibi. Human review is pending; personal learning answers remain unfilled.
