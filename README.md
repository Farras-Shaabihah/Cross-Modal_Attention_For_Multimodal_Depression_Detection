# Cross-Modal Attention for Multimodal Depression Detection Using Limited DAIC-WOZ Data

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Conference](https://img.shields.io/badge/ICITACEE-2025-blueviolet.svg)

> 📄 **Published at:** 12th International Conference on Information Technology, Computer, and Electrical Engineering (ICITACEE), 2025 — **Best Presenter Award**
>
> 🔗 **IEEE Xplore:** [View Paper](https://doi.org/10.1109/ICITACEE66165.2025.11232572)

---

## 📌 Overview

This repository contains the implementation code for a lightweight multimodal deep learning framework for **binary depression classification**. The model integrates **audio**, **visual**, and **textual** modalities from the DAIC-WOZ dataset using a **multi-head cross-modal attention fusion** mechanism.

Despite being trained on only **97 participants** due to storage constraints, the model achieves:

| Metric | Score |
|---|---|
| Accuracy | **80%** |
| Macro-averaged F1-score | **0.78** |
| Weighted F1-score | **0.81** |

---

## Model Architecture

The proposed architecture consists of three modality-specific subnetworks followed by a cross-modal attention fusion layer:

### Modality Subnetworks

| Modality | Feature Extraction | Subnetwork |
|---|---|---|
| **Audio** | MFCC, COVAREP, Formant features | SimpleRNN + Dense + Dropout + L2 |
| **Visual** | OpenFace (Action Units, Gaze, Head Pose) | Conv1D + MaxPooling + Dense |
| **Textual** | BERT-base embeddings | Dense + Dropout + BatchNorm |

### Cross-Modal Attention Fusion

After modality-specific feature extraction, each modality is projected into a **shared latent space**. A **multi-head cross-modal attention** mechanism then learns inter-modality dependencies by treating each modality as a query against the remaining two (e.g., Audio → Visual, Audio → Text, and all permutations).

The fusion outputs are concatenated, passed through **global average pooling**, and fed into a final classification head for binary prediction.

```
Audio Input ──► Audio Subnetwork ──► Projected Audio Emb. ──┐
Text Input  ──► Text Subnetwork  ──► Projected Text Emb.  ──┼──► Cross-Modal Attention ──► Classification
Visual Input──► Visual Subnetwork──► Projected Visual Emb.──┘
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Nadam (lr = 1e-5) |
| Batch Size | 64 |
| Max Epochs | 1,000 |
| Attention Heads | 2 (key dim = 16) |
| Dropout Rate | 0.3 |
| L2 Regularization | λ = 0.03 |
| Train/Test Split | 80:20 (stratified) |
| Loss Function | Sparse Categorical Cross-Entropy |
| Hardware | Google Colab (NVIDIA Tesla T4, 16GB) |

**Callbacks used:**
- `EarlyStopping` — patience of 10 epochs on validation loss
- `ReduceLROnPlateau` — factor 0.5 on validation loss plateau

**Class imbalance handling:**
- Manual minority oversampling
- SMOTE (Synthetic Minority Over-sampling Technique)

---

## Results

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Non-depressed | 0.62 | 0.83 | 0.71 | 6 |
| Depressed | 0.92 | 0.79 | 0.85 | 14 |
| **Accuracy** | | | **0.80** | 20 |
| Macro Avg | 0.77 | 0.81 | 0.78 | 20 |
| Weighted Avg | 0.83 | 0.80 | 0.81 | 20 |

The confusion matrix shows 5/6 non-depressed and 11/14 depressed samples correctly classified, with only 4 misclassifications total.

---

## Repository Structure

```
depression-detection/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   └── Binary_Class_Test_Model.ipynb                # Full training and evaluation notebook
│
└── results/
    ├── confusion_matrix.png
    └── training_curves.png
```

---

## Getting Started


## Requirements

> Full list available in `requirements.txt`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@INPROCEEDINGS{11232572,
  author={Shaabihah, Farras and Kusnawi, Kusnawi},
  booktitle={2025 12th International Conference on Information Technology, Computer, and Electrical Engineering (ICITACEE)}, 
  title={Cross-Modal Attention for Multimodal Depression Detection Using Limited DAIC-WOZ Data}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Visualization;Accuracy;Uncertainty;Computational modeling;Telemedicine;Mental health;Depression;Feature extraction;Robustness;Information technology;Depression detection;multimodal fusion;crossmodal attention;deep learning;DAIC-WOZ;mental health AI;intermediate fusion},
  doi={10.1109/ICITACEE66165.2025.11232572}
}
```

---

## ⚠️ Limitations

- Trained on a subset of **97 out of 189** available DAIC-WOZ participants due to storage constraints
- Binary classification only (depressed / non-depressed) — does not distinguish severity levels
- No uncertainty estimation integrated (planned for future work)

---

## Future Work

- Extend to **multiclass PHQ-8 severity classification** (mild, moderate, severe)
- Integrate **uncertainty quantification** for clinical reliability
- Evaluate generalizability on **external datasets**
- Incorporate **explainability techniques** (e.g., attention visualization)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

> Note: The DAIC-WOZ dataset is subject to its own licensing terms and is not covered by this license.

---

## Acknowledgements

- Dataset: DAIC-WOZ — developed through the [SimSensei project](https://dcapswoz.ict.usc.edu) by USC ICT
- Presented at **ICITACEE 2025** — awarded **Best Presenter**
- Department of Informatics, Faculty of Computer Science, Universitas Amikom Yogyakarta
