# MUTE Phase 2: Multimodal Stance Detection

**Multimodal Architecture for Stance Classification (Favour / Against / Neutral)** using **Google's MuRIL** (text) + **Google's ViT** (vision) with **Cross-Attention Fusion**.

This is Phase 2 of the MUTE project, focusing on fusing textual and visual features for improved stance detection on social media posts (likely images + captions).

## 📁 Dataset

The dataset (images + labels) can be downloaded from here:

[Google Drive Folder](https://drive.google.com/drive/folders/1zN2LMqu3W4kxe_DwDJhCgXoXrKFH3XYH?usp=drive_link)

- Place all **image files** and the `data.xlsx` (or `data.csv`) in the same directory as the Python scripts, **OR**
- Update the paths inside the Excel sheet / code accordingly.

The Excel file containing labels and metadata is already included in this repository.

## 🏗️ Architecture

- **Text Encoder**: [google/muril-base-cased](https://huggingface.co/google/muril-base-cased) — Multilingual Representations for Indian Languages (BERT-based, excellent for code-mixed and Indian language text).
- **Image Encoder**: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) (or ViT variant with patch size 8 — "ViT 244_8" likely refers to a ViT-B/8 224×224 model).
- **Fusion**: Cross-Attention Fusion mechanism to combine text and image embeddings.
- **Task**: 3-class classification — **Favour**, **Against**, **Neutral**.

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mute-phase-2.git
   cd mute-phase-2
