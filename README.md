# AI-Advisory-Hub-Voice-Enablement

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)
![TTS](https://img.shields.io/badge/Task-Text%20to%20Speech-orange.svg)

## Text-to-Speech Pipeline for Kenyan Agricultural Advisory

This repository implements a **VITS-based Text-to-Speech (TTS) system** for low-resource Kenyan languages: **Swahili, Kikuyu, and Kalenjin**, enabling natural voice delivery of agricultural advisories to smallholder farmers.

Swahili is used as a cross-lingual baseline, while Kikuyu is fine-tuned using the **Waxal TTS dataset** with agricultural domain adaptation. Kalenjin is incorporated using the Thiomi dataset.

---

## 📌 Key Features

- Neural TTS using **VITS (Variational Inference Text-to-Speech)**
- Multilingual support: Swahili, Kikuyu, Kalenjin
- Agricultural domain adaptation
- Mel-spectrogram based synthesis
- Low-resource language transfer learning
- Future mobile deployment support

---

## 📂 Repository Structure

```bash
TTS/
├── configs/
│   ├── mel_config.yaml
│   └── preprocessing.yaml
│
├── data/
│   ├── raw/
│   │   ├── waxal/
│   │   └── thiomi/
│   └── final/
│       └── processed/
│           ├── kikuyu/
│           └── swahili/
│
├── models/
│   └── checkpoints/
│
├── src/
│   ├── audio_processor.py
│   ├── data_utils.py
│   ├── project_config.py
│   └── text_processor.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_model_training.ipynb
│
├── requirements.txt
└── README.md
````

---

## 📊 Datasets

### Primary Dataset: Waxal TTS (Swahili & Kikuyu)

* Source: Google Research
* HuggingFace: [https://huggingface.co/datasets/google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP)
* Splits:

  * Kikuyu: `kik_tts`
  * Swahili: Swahili subset

### Secondary Dataset: Thiomi (Kalenjin)

* Used for Kalenjin TTS modeling

---

## ⚙️ Pipeline Overview

1. Data Exploration - dataset analysis
2. Preprocessing - noise removal, silence trimming, normalization
3. Feature Extraction - mel-spectrogram generation
4. Model Training - VITS fine-tuning on Kikuyu & Swahili

---

## 🧠 Model

Uses **VITS (Variational Inference Text-to-Speech)**:

* VAE-based latent modeling
* Flow-based generative alignment
* GAN discriminator for speech realism

---

## 📦 Installation

```bash
pip install torch torchaudio librosa soundfile pandas numpy matplotlib tqdm datasets pyyaml
```

---

## 🚀 Quick Start

### 1. Explore Data

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Preprocess Data

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### 3. Train Model

```bash
jupyter notebook notebooks/04_model_training.ipynb
```

---

## 📈 Work in Progress...

* Full Kalenjin dataset expansion
* XTTS-v2 zero-shot voice cloning integration
* Mobile/edge deployment optimization
* Real-time agricultural voice assistant

---

## 📚 Citations

### Waxal TTS Dataset

```bibtex
@dataset{waxal2023,
  title={Waxal TTS Dataset},
  author={Google Research},
  year={2023},
  url={https://huggingface.co/datasets/google/WaxalNLP}
}
```

### VITS Model

```bibtex
@inproceedings{kim2021vits,
  title={Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  booktitle={ICML},
  year={2021}
}
```
