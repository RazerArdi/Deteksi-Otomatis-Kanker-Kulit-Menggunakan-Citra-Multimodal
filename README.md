<div align="center">

# 🔬 MILKFusionNet

### *Comparative Framework: Classic ML vs. CNNs vs. Transformers with PEFT/LoRA*

![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge&logo=github)
![Framework](https://img.shields.io/badge/Framework-PyTorch_%7C_Scikit--Learn-EE4C2C?style=for-the-badge&logo=pytorch)
![Technique](https://img.shields.io/badge/Technique-LoRA_%7C_Feature_Eng-violet?style=for-the-badge&logo=huggingface)
![Dataset](https://img.shields.io/badge/Dataset-ISIC%20MILK--10k-00D4FF?style=for-the-badge&logo=kaggle)
![Language](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**[Dataset](https://challenge.isic-archive.com/data/#milk10k)** • **[Documentation](#-getting-started)** • **[Methodology](#-methodology)** • **[License](#-license)**

</div>

---

## 🎯 Overview

**MILKFusionNet** is a robust research framework designed to benchmark and analyze skin lesion classification strategies on the **ISIC MILK-10k** dataset. Moving beyond simple classification, this project rigorously compares three distinct paradigms:

1.  **Classic Machine Learning:** Interpretable models using hand-crafted feature engineering (RGB, GLCM, HOG, HSV, LBP).
2.  **Deep Learning (CNNs):** State-of-the-art convolutional networks adapted via **LoRA (Low-Rank Adaptation)**.
3.  **Vision Transformers:** Modern attention-based architectures adapted via **LoRA (PEFT)**.

> **Core Objective:** To evaluate whether modern Parameter-Efficient Fine-Tuning (PEFT) techniques on heavy architectures outperform traditional feature-based learning in scenarios with high class imbalance.

---

## 💡 The Challenge

Skin cancer, particularly melanoma, remains a critical global health concern with high mortality rates when diagnosed late. Current diagnostic paradigms face several limitations:

<table>
<tr>
<td width="33%" align="center">

### 🎭 Subjectivity
Heavy reliance on dermatologist experience with limited availability in underserved regions

</td>
<td width="33%" align="center">

### 📊 Data Imbalance
Severe class imbalance in medical datasets causing model bias toward common conditions

</td>
<td width="33%" align="center">

### 🔍 Unimodal Limitations
Most AI systems analyze single image types, missing contextual clinical information

</td>
</tr>
</table>

---

## 🏗️ Architecture & Workflows

The project is structured into three parallel experimental workflows:

```mermaid
graph TD;
    subgraph "Workflow A: Classic ML"
        A1[Input Image] --> B1[Preprocessing: Resize + CLAHE];
        B1 --> C1[Feature Extraction];
        C1 --> D1[30 Features: RGB, HSV, GLCM, HOG, LBP];
        D1 --> E1[Imbalance Handling: SMOTE / Class Weights];
        E1 --> F1[Models: SVC, RF, LogReg, KNN, GNB];
    end

    subgraph "Workflow B: CNNs (LoRA)"
        A2[Input Image] --> B2[Augmentation + Normalization];
        B2 --> C2[Backbone: ResNet50 / VGG16 / EfficientNet];
        C2 --> D2[Inject LoRA Adapters];
        D2 --> F2[Fine-Tuning (Low Rank)];
    end

    subgraph "Workflow C: Transformers (LoRA)"
        A3[Input Image] --> B3[Augmentation + Normalization];
        B3 --> C3[Backbone: ViT / Swin / MaxViT];
        C3 --> D3[Inject LoRA Adapters];
        D3 --> F3[Fine-Tuning (Low Rank)];
    end

    F1 & F2 & F3 --> G[🏆 Grand Final Analysis];
    G --> H[Confusion Matrix & ECE Calibration];
```

---

## 🛠️ Methodology

### 1. Workflow A: Classic Feature Engineering 🧑‍🔬

Instead of raw pixels, we extract interpretable dermoscopic features to train lightweight classifiers.

| Feature Set | Count | Description |
|:---|:---:|:---|
| **Color (RGB)** | 12 | Mean, Std, Skew, Kurtosis per channel |
| **Color (HSV)** | 6 | Mean, Std for Hue, Saturation, Value |
| **Texture (GLCM)** | 6 | Contrast, Correlation, Energy, Homogeneity, ASM, Dissimilarity |
| **Texture (LBP)** | 3 | Local Binary Patterns (Micro-texture) statistics |
| **Shape (HOG)** | 3 | Histogram of Oriented Gradients statistics |
| **Total** | **30** | Standardized features |

* **Handling Imbalance:** Comparative study between `Class Weights` and `SMOTE` (Synthetic Minority Over-sampling).

### 2. Workflows B & C: Deep Learning with PEFT 🚀

We utilize **LoRA (Low-Rank Adaptation)** to fine-tune massive pre-trained models efficiently. By freezing the pre-trained backbone and only training rank-decomposition matrices, we reduce trainable parameters by >90% while preventing catastrophic forgetting.

| Architecture Type | Models Evaluated | Technique |
|:---|:---|:---|
| **CNN** | ResNet-50, VGG-16, EfficientNet-B0 | LoRA on Conv/Linear layers |
| **Transformer** | ViT-B/16, Swin-T, MaxViT-T | LoRA on Attention (qkv) & MLP |

**Training Config:**

* **Optimizer:** Adam (CNN) / AdamW (Transformer)
* **Loss:** CrossEntropyLoss with Class Weights
* **Pipeline:** Albumentations (Flip, Rotate, ColorJitter, CoarseDropout)

---

## 📊 Analysis Framework

This project goes beyond accuracy, implementing a "PhD-level" analysis suite:

### 1. Champion vs. Champion

A head-to-head comparison of the best model from each category (e.g., *SVC vs. EfficientNet vs. MaxViT*).

### 2. Reliability Diagrams (Calibration)

We calculate the **Expected Calibration Error (ECE)** to measure model confidence.

> *Does a 90% confidence prediction actually mean the model is right 90% of the time?*

### 3. Error Analysis

* **Top Confused Pairs:** Quantifying which classes are most frequently mistaken (e.g., Melanoma vs. Nevus).
* **High Confidence Failures:** Visualizing images where the model was "Wrong but Certain".

---

## 📊 Dataset Overview

### ISIC MILK-10k Dataset

The **MILK-10k (Metadata-Informed Lesion Knowledge)** dataset is a comprehensive multimodal collection for skin lesion classification, containing over 10,000 cases with rich metadata and dual imaging modalities.

<details>
<summary><b>🗂️ Dataset Statistics & Distribution</b></summary>

<br>

#### Dataset Composition

| Split | Total Cases | Dermoscopic Images | Clinical Images | Metadata Records |
|:------|:------------|:-------------------|:----------------|:-----------------|
| **Training** | ~8,000 | ✅ Available | ✅ Available | ✅ Complete |
| **Testing** | ~2,000 | ✅ Available | ✅ Available | ✅ Complete |
| **Total** | **~10,000** | **10,000+** | **10,000+** | **10,000+** |

</details>

<details>
<summary><b>🎯 Class Distribution (11 Diagnostic Categories)</b></summary>

<br>

The dataset includes 11 distinct skin lesion types with varying prevalence:

| Class | Diagnosis | Abbr. | Approx. Samples | Severity | Description |
|:-----:|:----------|:------|:----------------|:---------|:------------|
| **0** | Actinic Keratosis | AK | ~600 | ⚠️ Pre-cancerous | Rough, scaly patches from sun exposure |
| **1** | Basal Cell Carcinoma | BCC | ~900 | 🔴 Malignant | Most common skin cancer, slow-growing |
| **2** | Benign Keratosis | BKL | ~1,200 | 🟢 Benign | Harmless skin growths (seborrheic keratosis) |
| **3** | Dermatofibroma | DF | ~400 | 🟢 Benign | Firm nodular skin lesions |
| **4** | Melanoma | MEL | ~800 | 🔴 Highly Malignant | Deadliest skin cancer, requires urgent care |
| **5** | Melanocytic Nevus | NV | ~3,500 | 🟢 Benign | Common moles, most prevalent class |
| **6** | Squamous Cell Carcinoma | SCC | ~700 | 🔴 Malignant | Second most common skin cancer |
| **7** | Vascular Lesion | VASC | ~350 | 🟢 Benign | Blood vessel abnormalities (hemangiomas) |
| **8** | Acral Lentiginous Melanoma | ALM | ~150 | 🔴 Highly Malignant | Rare melanoma subtype on extremities |
| **9** | Lentigo Maligna | LM | ~200 | ⚠️ Pre-cancerous | Early melanoma on sun-damaged skin |
| **10** | Merkel Cell Carcinoma | MCC | ~50 | 🔴 Highly Malignant | Rare aggressive neuroendocrine tumor |

**Class Imbalance Ratio:** ~70:1 (Most common: NV | Rarest: MCC)

> 📌 **Note:** This severe imbalance motivates our use of Focal Loss and ensemble strategies to prevent model bias toward common benign lesions.

</details>

<details>
<summary><b>🖼️ Image Modalities & Specifications</b></summary>

<br>

#### Dermoscopic Images
- **Purpose:** High-magnification surface analysis revealing subsurface structures
- **Equipment:** Dermatoscope with polarized/non-polarized light
- **Resolution:** Variable (typically 1024×768 to 6000×4000 pixels)
- **Format:** JPEG
- **Key Features:** Reveals pigment networks, globules, streaks, blue-white veil
- **Clinical Value:** Gold standard for melanoma screening

#### Clinical Images  
- **Purpose:** Contextual macro-view of lesion and surrounding skin
- **Equipment:** Standard digital camera
- **Resolution:** Variable (typically 1024×768 to 4000×3000 pixels)
- **Format:** JPEG
- **Key Features:** Shows lesion size, borders, surrounding erythema
- **Clinical Value:** Provides anatomical context and scale reference

#### Image Characteristics
| Aspect | Dermoscopic | Clinical |
|:-------|:------------|:---------|
| **Magnification** | 10-70× | 1× (macro) |
| **Field of View** | Lesion-focused | Wide anatomical context |
| **Lighting** | Controlled polarized | Natural/flash |
| **Diagnostic Use** | Structural analysis | Overall assessment |

</details>

<details>
<summary><b>📋 Tabular Metadata Attributes</b></summary>

<br>

The dataset includes comprehensive patient and lesion metadata:

#### Patient Demographics
| Attribute | Type | Description | Example Values |
|:----------|:-----|:------------|:---------------|
| `age_approx` | Integer | Patient age (years) | 25, 45, 67, 82 |
| `sex` | Categorical | Biological sex | male, female |
| `anatom_site_general` | Categorical | Body location | torso, lower extremity, upper extremity, head/neck, palms/soles, oral/genital |

#### Lesion Characteristics
| Attribute | Type | Description | Clinical Significance |
|:----------|:-----|:------------|:---------------------|
| `tbp_lv_A` | Float | Total body photography lesion area | Size indicator |
| `tbp_lv_Aext` | Float | Extended lesion area | Growth assessment |
| `tbp_lv_B` | Float | Border irregularity score | Higher = more irregular |
| `tbp_lv_C` | Float | Color variation index | Multicolor lesions |
| `tbp_lv_H` | Float | Hue homogeneity | Color distribution |
| `tbp_lv_Hext` | Float | Extended hue metrics | Advanced color analysis |
| `tbp_lv_L` | Float | Lightness/luminance | Brightness assessment |
| `tbp_lv_Lext` | Float | Extended lightness | Pigmentation depth |
| `tbp_lv_nevi_confidence` | Float | Confidence score for nevus | AI pre-screening score |
| `tbp_lv_norm_border` | Float | Normalized border score | Standardized irregularity |
| `tbp_lv_norm_color` | Float | Normalized color variance | Standardized color metric |
| `tbp_lv_perimeterMM` | Float | Lesion perimeter (mm) | Physical boundary size |
| `tbp_lv_areaMM2` | Float | Lesion area (mm²) | Physical size measurement |
| `tbp_lv_x` | Integer | X-coordinate on body map | Spatial localization |
| `tbp_lv_y` | Integer | Y-coordinate on body map | Spatial localization |
| `tbp_lv_z` | Integer | Z-coordinate (depth) | 3D positioning |

#### Derived Features
| Feature | Calculation | Clinical Relevance |
|:--------|:------------|:-------------------|
| `asymmetry_score` | Computed from A/B metrics | ABCDE rule (Asymmetry) |
| `border_score` | `tbp_lv_B + tbp_lv_norm_border` | ABCDE rule (Border) |
| `color_diversity` | `tbp_lv_C + tbp_lv_norm_color` | ABCDE rule (Color) |
| `diameter_mm` | `√(areaMM2/π) × 2` | ABCDE rule (Diameter) |

> 🏥 **ABCDE Rule:** Clinical mnemonic for melanoma detection (Asymmetry, Border, Color, Diameter, Evolution)

#### Missing Data Handling
- **Age:** Imputed with median age by sex and diagnosis
- **Location:** Imputed with mode (most common anatomical site)
- **Numeric Features:** KNN imputation (k=5) based on similar lesions
- **Missing Rate:** <5% for most attributes, <15% overall

</details>

<details>
<summary><b>🔬 Data Quality & Preprocessing Pipeline</b></summary>

<br>

#### Quality Assurance
- ✅ All images manually reviewed by certified dermatologists
- ✅ Multiple expert consensus for melanoma cases
- ✅ Standardized imaging protocols across collection sites
- ✅ Duplicate detection and removal
- ✅ Artifact filtering (rulers, hair, bubbles)

#### Preprocessing Steps
```python
# Image Pipeline
1. Resize to 224×224 pixels (efficient computation)
2. CLAHE enhancement (clip_limit=3.0)
3. Hair removal algorithm (optional)
4. Color normalization (Reinhard method)
5. Augmentation (rotation, flip, color jitter)
6. Normalization (ImageNet statistics)

# Metadata Pipeline  
1. Outlier detection (IQR method)
2. Missing value imputation
3. Feature scaling (StandardScaler)
4. Categorical encoding (Label/One-Hot)
5. Feature engineering (ABCDE scores)
```

#### Data Splits
- **Stratified Split:** Maintains class distribution across train/val/test
- **Cross-Validation:** 5-fold stratified for robust evaluation
- **Patient-Level Split:** No data leakage (same patient not in train/test)

</details>

---

## 📂 Project Structure

```
📦 MILKFusionNet/
│
├── 📂 dataset/                  # Raw ISIC MILK-10k data
│
├── 📂 notebooks/
│   └── 📄 main.ipynb            # Single end-to-end pipeline notebook
│
├── 📂 processed_data/           # Cached CSVs for extracted features
│   ├── 📄 train_color_features.csv
│   ├── 📄 train_glcm_features.csv
│   └── ...
│
├── 📂 src/                      # (Optional) Modularized scripts
│
├── 📄 requirements.txt          # Dependencies (torch, peft, sklearn, etc.)
└── 📄 README.md                 # This documentation
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* CUDA-capable GPU (Recommended for LoRA training)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/username/MILKFusionNet.git

# 2. Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas numpy matplotlib seaborn tqdm opencv-python

# 3. Install specialized libraries for PEFT and Metrics
pip install peft torchmetrics albumentations imbalanced-learn
```

### Running the Pipeline

The entire workflow is encapsulated in `main.ipynb`. Run the cells sequentially to:

1.  Perform EDA and Class Distribution Analysis.
2.  Extract features and train Classic ML models (SVC, RF, etc.).
3.  Fine-tune CNNs (ResNet, VGG) and Transformers (ViT, Swin) using LoRA.
4.  Generate the Grand Final Leaderboard and Analysis plots.

---

## 📊 Expected Results

| Metric | Target | Clinical Significance |
|:-------|:------:|:---------------------|
| **Accuracy** | >85% | Overall diagnostic precision |
| **Balanced Accuracy** | >80% | Performance across rare classes |
| **Sensitivity (Melanoma)** | >90% | Critical for early cancer detection |
| **Specificity** | >88% | Reduces false positive burden |

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍🔬 Author

**Bayu Ardiyansyah**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github)](https://github.com/username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/username)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

<div align="center">

> **IMPORTANT:** MILKFusionNet is a research prototype and **NOT** a certified medical diagnostic tool. 
> 
> All predictions must be validated by qualified healthcare professionals. This system is designed to augment, not replace, clinical expertise.

</div>

---

<div align="center">

### 🌟 If you find this project useful, please consider giving it a star!

**Made with ❤️ for advancing healthcare AI**

</div>