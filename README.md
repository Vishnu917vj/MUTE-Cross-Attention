<img width="91" height="150" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/1ee77566-c980-4f88-aeb3-3dac4cbab116" /># MUTE Phase 2: Multimodal Stance Detection

**Multimodal Architecture for Stance Classification (Favour / Against / Neutral)**  
using **Google's MuRIL** (text) + **Google's ViT** (vision) with **Cross-Attention Fusion**.

This is **Phase 2** of the MUTE project, focusing on fusing textual and visual features for improved stance detection on social media posts (images + captions).

## 📁 Dataset

The dataset (images + labels) can be downloaded from here:

[Google Drive Folder](https://drive.google.com/drive/folders/1zN2LMqu3W4kxe_DwDJhCgXoXrKFH3XYH?usp=drive_link)

- Place all **image files** and the `data.xlsx` (or `data.csv`) in the root directory of the project.
- Alternatively, update the paths in the code or Excel sheet as needed.

> **Note**: The Excel file containing labels and metadata is already included in this repository.

## 🏗️ Architecture

- **Text Encoder**: [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased) — Multilingual Representations for Indian Languages (BERT-based, excellent for code-mixed and Indian language text).
- **Image Encoder**: [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224) (ViT-B/16). Experiments were also done with patch size 8.
- **Fusion**: Cross-Attention mechanism to combine text and image embeddings.
- **Task**: 3-class classification — **Favour**, **Against**, **Neutral**.

## 🧠 Model Architecture

The model uses a **late fusion** strategy with **Cross-Attention** to effectively combine textual and visual information for stance classification.

### 1. Text Encoder
- **Model**: [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased)
- **Type**: BERT-based Transformer
- **Details**: Pre-trained on 17 Indian languages + transliterated versions. Highly effective for code-mixed Indian social media text.
- **Output**: 768-dimensional embedding (from `[CLS]` token or mean pooling).

### 2. Image Encoder
- **Model**: Google **Vision Transformer (ViT)**
- **Variant**: `vit-base-patch16-224` (224×224 resolution)
- **Details**: Images are split into fixed-size patches, linearly embedded, and processed by a Transformer encoder. Pre-trained on ImageNet-21k.
- **Output**: 768-dimensional embedding (from `[CLS]` token).

### 3. Cross-Attention Fusion Module
After extracting features from both modalities, a **Cross-Attention** mechanism is applied:
- Text features act as **Query** to attend to Image features (**Key** & **Value**), or
- Bidirectional cross-attention (text ↔ image).
- Multiple multi-head cross-attention layers capture fine-grained interactions between text and visual content.
- The fused representation then passes through:
  - Layer Normalization
  - Feed-Forward Network (MLP)
  - Dropout

### 4. Classification Head
- A final linear layer maps the fused embedding to **3 output classes**: Favour, Against, Neutral.
- **Loss**: Cross-Entropy Loss
- **Optimizer**: AdamW

## 🚀 Quick Start

## clone 
 - git clone https://github.com/Vishnu917vj/MUTE-Cross-Attention.git
 - cd MUTE-Cross-Attention
 # Install PyTorch (choose the correct CUDA version for your system)
  - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

  # Install other dependencies
  - pip install transformers datasets pandas openpyxl tqdm
  - python train_cross_attention.py
  
##  architecture
![Demo Image](MUTE-Cross-Attention/mermaid-diagram.svg)


The model is implemented in PyTorch using the Hugging Face Transformers library.
Trained models are saved as .pth files in the models/ directory.
You can easily modify the number of cross-attention layers, hidden dimensions, or fusion strategy in model.py.
