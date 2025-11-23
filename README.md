<div align="center">

# 🔬 MILKFusionNet
### *Comparative Framework: Classic ML vs. CNNs vs. Transformers with PEFT/LoRA*

![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge&logo=github)
![Framework](https://img.shields.io/badge/Framework-PyTorch_%7C_Scikit--Learn-EE4C2C?style=for-the-badge&logo=pytorch)
![Technique](https://img.shields.io/badge/Technique-LoRA_%7C_Imbalance_Handling-violet?style=for-the-badge&logo=huggingface)
![Dataset](https://img.shields.io/badge/Dataset-ISIC%20MILK--10k-00D4FF?style=for-the-badge&logo=kaggle)
![Language](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**[Dataset](https://challenge.isic-archive.com/data/#milk10k)** • **[Methodology](#-methodology)** • **[Results](#-results--benchmarking)** • **[License](#-license)**

</div>

---

## 🎯 Overview

**MILKFusionNet** is a comprehensive research framework designed to benchmark skin lesion classification strategies on the **ISIC MILK-10k** dataset (11 Diagnostic Categories). Moving beyond simple binary classification, this project rigorously evaluates **22 model variations** across three distinct paradigms:

1.  **Classic Machine Learning:** Interpretable baselines using hand-crafted feature engineering (Color, Texture, Shape).
2.  **Deep Learning (CNNs):** Benchmarking Custom architectures vs. Industry-standard backbones (ResNet, VGG, EfficientNet).
3.  **Vision Transformers:** Evaluating State-of-the-Art attention mechanisms (ViT, Swin, DeiT).

> **Core Innovation:** The application of **Low-Rank Adaptation (LoRA)** on both CNN and Transformer architectures to demonstrate that **>99% parameter efficiency** can be achieved without compromising diagnostic accuracy on imbalanced medical datasets.

---

## 🏗️ Architecture & Workflows

The project executes three parallel experimental workflows:

```mermaid
graph TD;
    subgraph "Workflow A: Classic ML (Baseline)"
        A1[Input Image] --> B1[Feature Extraction];
        B1 --> C1["Features: RGB, HSV, GLCM, HOG, LBP"];
        C1 --> D1["Imbalance: SMOTE / Class Weights"];
        D1 --> E1["Models: SVM, Random Forest, XGBoost"];
    end

    subgraph "Workflow B: CNNs (Deep Learning)"
        A2["Input Image (224x224)"] --> B2["Augmentation + Normalization"];
        B2 --> C2["Architectures: CustomCNN, ResNet50, VGG16, EffNetB0"];
        C2 --> D2["Strategies: Scratch, Frozen, Full-FT, LoRA"];
        D2 --> E2["Training (Weighted Loss)"];
    end

    subgraph "Workflow C: Vision Transformers"
        A3["Input Image (224x224)"] --> B3["Augmentation + Normalization"];
        B3 --> C3["Architectures: ViT-Base, Swin-Tiny, DeiT-Base"];
        C3 --> D3["Strategies: Scratch, Full-FT, LoRA"];
        D3 --> E3["Training (Weighted Loss)"];
    end

    E1 & E2 & E3 --> F["🏆 Grand Final Analysis"];
    F --> G["Efficiency Plots (Acc vs Params) & Dual-Lang Posters"];
````

-----

## 🛠️ Methodology

### 1\. Dataset & Preprocessing 🖼️

  * **Dataset:** ISIC Archive MILK-10k (11 Classes: Melanoma, Nevus, BCC, etc.).
  * **Preprocessing:** Resize (224x224), Normalization (ImageNet Mean/Std).
  * **Augmentation:** Random Horizontal/Vertical Flip, Rotation (20°), Color Jitter.
  * **Imbalance Handling:**
      * *Classic ML:* SMOTE Oversampling.
      * *Deep Learning:* **Weighted Cross-Entropy Loss** (Calculated based on inverse class frequency).

### 2\. Model Configurations 🤖

We evaluated a total of **22 model configurations**:

### 2. Model Configurations 🤖

We evaluated a total of **22 model configurations** across three distinct paradigms:

| Category | Architecture | Training Strategies | Technical Details |
|:---|:---|:---|:---|
| **Classic ML** | **SVM, Random Forest, XGBoost** | Feature Engineering | Inputs: GLCM, HOG, Color Moments (30 Features) |
| **CNN** | **CustomVanillaCNN** | Scratch | Lightweight Baseline (3 Conv Blocks) |
| **CNN** | **ResNet50** | Frozen, Full FT, **LoRA** | LoRA Target: `layerX.conv2` |
| **CNN** | **VGG16** | Frozen, Full FT, **LoRA** | LoRA Target: `features` blocks |
| **CNN** | **EfficientNet-B0** | Frozen, Full FT, **LoRA** | LoRA Target: `MBConv` expansion layers |
| **Transformer** | **ViT-Base** | Scratch, Full FT, **LoRA** | LoRA Target: `query`, `value` |
| **Transformer** | **Swin-Tiny** | Scratch, Full FT, **LoRA** | Hierarchical Window Attention |
| **Transformer** | **DeiT-Base** | Scratch, Full FT, **LoRA** | Distilled Knowledge (Replaces MaxViT) |

### 3\. Training Setup ⚙️

  * **Framework:** PyTorch, Hugging Face `transformers`, `peft`.
  * **Optimizer:** `AdamW` (Weight Decay 0.01).
  * **Learning Rate:** `1e-4` (CNN), `2e-5` (Transformers).
  * **Epochs:** 10 (with validation per epoch).
  * **Batch Size:** 32 (Optimized for stability).
  * **Memory Management:** Automated CPU-offloading post-training to prevent GPU OOM.

-----

## 📊 Results & Benchmarking

The pipeline automatically generates **High-DPI** (300 DPI) visualizations stored in `notebook/image/`, featuring dual-language support (English & Indonesian).

### 1\. Efficiency Analysis

We performed a trade-off analysis between Accuracy and Model Size:

  * **Scatter Plot:** Visualizes models in the "High Accuracy - Low Params" quadrant.
  * **Bar Chart (Log Scale):** Demonstrates the extreme parameter reduction achieved by LoRA (from \~86 Million to \~300 Thousand trainable parameters).

### 2\. Key Visualizations

  * **Training Dynamics:** Loss/Accuracy curves per epoch comparing DL models against the Classic ML baseline.
  * **Confusion Matrix:** Heatmaps identifying specific inter-class confusion (e.g., differentiating *Melanoma* from *Nevus*).
  * **Leaderboard:** Horizontal bar charts ranking all 22 models based on Validation Accuracy.

-----

## 📂 Project Structure

The repository is structured as follows:

```bash
MILKFusionNet/
├── 📂 dataset/                  # Raw ISIC data (Train/Test splits)
│
├── 📂 Models/                   # 💾 Saved Model Weights (.pth)
│   ├── CustomCNN_Scratch.pth
│   ├── ResNet50_LoRA.pth
│   ├── DeiT_LoRA.pth
│   └── ... (All 22 models)
│
├── 📂 notebook/                 # 📓 Main Experiment Environment
│   ├── main.ipynb               # End-to-End Pipeline Code
│   ├── poster_table_results.csv # Raw results data
│   │
│   └── 📂 image/                # 🖼️ GENERATED OUTPUTS (High-Res)
│       ├── all_model_results.csv
│       ├── Fig1_Data_Distribution_English.png
│       ├── Fig2_Training_Dynamics_English.png
│       ├── Fig5_Parameter_Efficiency_English.png
│       ├── Fig6_Accuracy_vs_Params_English.png
│       └── ... (Confusion Matrices for every model)
│
├── 📂 processed_data/           # Intermediate files (Efficiency stats)
├── 📄 requirements.txt          # Project dependencies
└── 📄 README.md                 # This documentation
```

-----

## 🚀 Getting Started

### Prerequisites

Ensure your environment supports CUDA (GPU recommended for Transformers).

```bash
# Install core libraries
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install specific research tools
pip install transformers peft scikit-learn pandas matplotlib seaborn opencv-python tqdm imbalanced-learn
```

### Execution Guide

Run the `notebook/main.ipynb` cells sequentially:

1.  **Data Prep:** Load dataset, perform stratified splitting, and apply augmentation.
2.  **Workflow A:** Feature extraction & Classic ML training.
3.  **Workflow B:** CNN Factory setup & Training Loop (10 Models).
4.  **Workflow C:** Transformer Factory setup & Training Loop (9 Models).
5.  **Grand Final:** Execute the final visualization cells to generate posters and CSVs.

-----

## 👨‍🔬 Author

**Bayu Ardiyansyah**

  * **Focus:** Medical Image Analysis, Deep Learning, Computer Vision.
  * **Tech Stack:** PyTorch, Scikit-Learn, Hugging Face.

-----

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

<div align="center"\>

> **Disclaimer:** MILKFusionNet is a research prototype. Predictions should not replace professional medical diagnosis.

**⭐ Star this repository if you find it useful for your research\!**