[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gesturelsm-latent-shortcut-based-co-speech/gesture-generation-on-beat2)](https://paperswithcode.com/sota/gesture-generation-on-beat2?p=gesturelsm-latent-shortcut-based-co-speech) <a href="https://arxiv.org/abs/2501.18898"><img src="https://img.shields.io/badge/arxiv-gray?logo=arxiv&amp"></a>

## GestureLSM: Latent Shortcut based Co-Speech Gesture Generation with Spatial-Temporal Modeling  
**ICCV 2025**

GestureLSM is a co-speech gesture generation framework that leverages diffusion models, latent shortcut modeling, and MeanFlow with spatial-temporal reasoning. This repository contains everything needed for training, testing, and running demos, as well as pretrained models and configuration files.

---

## 1. Project Status & Highlights

### 1.1 Release Checklist

- [x] Inference code
- [x] Pretrained models
- [x] Web-based demo
- [x] Training scripts
- [x] Refactored, cleaner codebase
- [x] Support for [MeanFlow](https://arxiv.org/abs/2505.13447)
- [x] Unified training and testing pipeline
- [ ] MeanFlow training code (planned, coming soon)
- [ ] Integration with [Intentional-Gesture](https://github.com/andypinxinliu/Intentional-Gesture)

### 1.2 Recent Code Updates

- The repository has been **cleaned and reorganized**; older versions are preserved in the `old` branch for historical reference.
- **New capabilities**:
  - MeanFlow model support
  - A single `train.py` entry point that now handles both training and evaluation
  - New configuration files in `configs_new/`
  - Updated checkpoints with improved performance

---

## 2. Quick Start

### 2.1 Demo / Inference Without Dataset

Run a web-based demo (no dataset needed):

```bash
# Run the web demo with the Shortcut model
python demo.py -c configs/shortcut_rvqvae_128_hf.yaml
```

### 2.2 Testing with Your Own Audio/Text

```bash
# Test with your own audio and text (requires pretrained models)
python train.py --config configs_new/shortcut_rvqvae_128.yaml --ckpt ckpt/shortcut_reflow.bin --mode test
```

The demo provides:

- A browser-based interface
- Near real-time gesture generation
- Support for custom audio and text input
- Visualization of the generated gestures

---

## 3. Installation & Environment

### 3.1 Environment Setup

Use the following commands to create and configure the environment:

```bash
conda create -n gesturelsm python=3.12
conda activate gesturelsm
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
bash demo/install_mfa.sh
```

---

## 4. Repository Structure

Understanding the layout will help you customize and extend the project more easily.

```text
GestureLSM/
├── configs_new/                      # New, unified configuration files
│   ├── diffusion_rvqvae_128.yaml    # Diffusion model config
│   ├── shortcut_rvqvae_128.yaml     # Shortcut model config
│   └── meanflow_rvqvae_128.yaml     # MeanFlow model config
├── configs/                          # Legacy configuration files (deprecated)
├── ckpt/                             # Pretrained model checkpoints
│   ├── new_540_diffusion.bin        # Diffusion model weights
│   ├── shortcut_reflow.bin          # Shortcut model weights
│   ├── meanflow.pth                 # MeanFlow model weights
│   └── net_300000_*.pth            # RVQ-VAE model weights
├── models/                           # Model implementations
│   ├── Diffusion.py                 # Diffusion model
│   ├── LSM.py                       # Latent Shortcut Model
│   ├── MeanFlow.py                  # MeanFlow model
│   ├── layers/                      # Neural network layers
│   ├── vq/                          # Vector quantization modules
│   └── utils/                       # Model utilities
├── dataloaders/                      # Data loading and preprocessing
│   ├── beat_sep_lower.py            # Main dataset loader
│   ├── pymo/                        # Motion processing library
│   └── utils/                       # Data utilities
├── trainer/                          # Training framework
│   ├── base_trainer.py              # Base trainer class
│   └── generative_trainer.py        # Generative model trainer
├── utils/                            # General utilities
│   ├── config.py                    # Configuration management
│   ├── metric.py                    # Evaluation metrics
│   └── rotation_conversions.py      # Rotation utilities
├── demo/                             # Demo and visualization
│   ├── examples/                    # Sample audio files
│   └── install_mfa.sh               # MFA installation script
├── datasets/                         # Dataset storage
│   ├── BEAT_SMPL/                   # Original BEAT dataset
│   ├── beat_cache/                  # Preprocessed cache
│   └── hub/                         # SMPL models and pretrained weights
├── outputs/                          # Training outputs and logs
│   └── weights/                     # Saved model weights
├── train.py                         # Unified training/testing script
├── demo.py                          # Web demo script
├── rvq_beatx_train.py               # RVQ-VAE training script
└── requirements.txt                 # Python dependencies
```

---

## 5. Core Components

### 5.1 Model Architecture

- **`models/Diffusion.py`** – Denoising diffusion model for high-quality gesture generation  
- **`models/LSM.py`** – Latent Shortcut Model for efficient, fast inference  
- **`models/MeanFlow.py`** – Flow-based, single-step MeanFlow model  
- **`models/vq/`** – Vector quantization modules for latent-space compression  

### 5.2 Configuration System

- **`configs_new/`** – Modern, unified configurations for all models  
- **`configs/`** – Older configuration files (kept for backward compatibility)  
- Each YAML config specifies model hyperparameters, training settings, and data paths.

### 5.3 Data Pipeline

- **`dataloaders/beat_sep_lower.py`** – Primary loader for the BEAT dataset  
- **`dataloaders/pymo/`** – Motion processing utilities for gesture data  
- **`datasets/beat_cache/`** – Preprocessed cache for faster data loading  

### 5.4 Training Framework

- **`train.py`** – Single script used to train and test all models  
- **`trainer/`** – Implements the base and generative trainer abstractions  
- **`optimizers/`** – Optimizer and scheduler definitions used by the trainer  

### 5.5 Utility Modules

- **`utils/config.py`** – Configuration parsing, management, and validation  
- **`utils/metric.py`** – Evaluation metrics (including FGD and related measures)  
- **`utils/rotation_conversions.py`** – 3D rotation conversion utilities  

---

## 6. Models & Checkpoints

### 6.1 Download Pretrained Models

```bash
# Option 1: From Google Drive
# Download the pretrained models (Diffusion + Shortcut + MeanFlow + RVQ-VAEs)
gdown https://drive.google.com/drive/folders/1OfYWWJbaXal6q7LttQlYKWAy0KTwkPRw?usp=drive_link -O ./ckpt --folder

# Option 2: From Hugging Face Hub
huggingface-cli download https://huggingface.co/pliu23/GestureLSM --local-dir ./ckpt

# Download the SMPL model
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8?usp=drive_link -O ./datasets/hub --folder
```

**Available Checkpoints**

- **Diffusion Model**: `ckpt/new_540_diffusion.bin`  
- **Shortcut Model**: `ckpt/shortcut_reflow.bin`  
- **MeanFlow Model**: `ckpt/meanflow.pth`  
- **RVQ-VAE Models**:  
  - `ckpt/net_300000_upper.pth`  
  - `ckpt/net_300000_hands.pth`  
  - `ckpt/net_300000_lower.pth`

---

## 7. Dataset

> Required for **training** and **evaluation**.  
> Not required for running the web demo or simple inference.

### 7.1 BEAT2 Dataset from Hugging Face

The original download script is no longer functional. Please use the Hugging Face version:

```bash
# Download BEAT2 dataset from Hugging Face
huggingface-cli download H-Liu1997/BEAT2 --local-dir ./datasets/BEAT2
```

**Dataset Information**

- **Source**: [H-Liu1997/BEAT2 on Hugging Face](https://huggingface.co/datasets/H-Liu1997/BEAT2)  
- **Size**: ~4.1K samples  
- **Format**: CSV with train/test splits  
- **License**: Apache 2.0  

### 7.2 Legacy Download (Deprecated)

```bash
# This command is deprecated and no longer works
# bash preprocess/bash_raw_cospeech_download.sh
```

---

## 8. Training

> Dataset download is required before training.

### 8.1 Unified Training Pipeline

Training is handled through a unified `train.py` script and the configs in `configs_new/`:

```bash
# Train Diffusion Model
python train.py --config configs_new/diffusion_rvqvae_128.yaml

# Train Shortcut Model
python train.py --config configs_new/shortcut_rvqvae_128.yaml

# Train MeanFlow Model
python train.py --config configs_new/meanflow_rvqvae_128.yaml
```

**Training Configuration Notes**

- **Config Directory**: Use `configs_new/` for up-to-date configs  
- **Output Directory**: Models are saved to `./outputs/weights/`  
- **Logging**: Supports Weights & Biases (configure in YAML files)  
- **GPU Usage**: Set devices and related options in the config files  

### 8.2 Legacy Training (Deprecated)

```bash
# Old training commands (deprecated)
python train.py -c configs/shortcut_rvqvae_128.yaml
python train.py -c configs/diffuser_rvqvae_128.yaml
```

### 8.3 Training RVQ-VAEs (1-Speaker)

> Requires the dataset.

```bash
bash train_rvq.sh
```

---

## 9. Testing & Evaluation

> **Note**: The dataset is required for evaluation.  
> For inference-only use cases, see the Quick Start section.

### 9.1 Unified Testing Pipeline

Use the unified `train.py` script with `--mode test`:

```bash
# Test Diffusion Model (20 steps)
python train.py --config configs_new/diffusion_rvqvae_128.yaml --ckpt ckpt/new_540_diffusion.bin --mode test

# Test Shortcut Model (2-step reflow)
python train.py --config configs_new/shortcut_rvqvae_128.yaml --ckpt ckpt/shortcut_reflow.bin --mode test

# Test MeanFlow Model (1-step flow-based)
python train.py --config configs_new/meanflow_rvqvae_128.yaml --ckpt ckpt/meanflow.pth --mode test
```

### 9.2 Legacy Testing (Deprecated)

```bash
# Old testing commands (deprecated)
python test.py -c configs/shortcut_rvqvae_128.yaml
python test.py -c configs/shortcut_reflow_test.yaml  
python test.py -c configs/diffuser_rvqvae_128.yaml
```

### 9.3 Model Comparison

| Model       | Steps | Description               | Key Features                     | Use Case                      |
|------------|-------|---------------------------|----------------------------------|-------------------------------|
| **Diffusion** | 20    | Denoising diffusion model | High quality, slower inference   | High-quality generation       |
| **Shortcut**  | 2–4   | Latent shortcut + reflow  | Fast inference, good quality     | **Recommended for most users** |
| **MeanFlow**  | 1     | Flow-based generation     | Fastest, single-step inference   | Real-time applications        |

### 9.4 Performance Comparison

| Model             | Steps | FGD Score ↓ | Beat Constancy ↑ | L1Div Score ↓ | Inference Speed |
|------------------|-------|-------------|------------------|---------------|-----------------|
| **MeanFlow**     | 1     | **0.4031**  | **0.7489**       | 12.4631       | **Fastest**     |
| **Diffusion**    | 20    | 0.4100      | 0.7384           | 12.5752       | Slowest         |
| **Shortcut**     | 20    | 0.4040      | 0.7144           | 13.4874       | Fast            |
| **Shortcut-ReFlow** | 2  | 0.4104      | 0.7182           | **13.678**    | Fast            |

**Legend**

- **FGD Score (↓)**: Lower → better gesture quality  
- **Beat Constancy (↑)**: Higher → better audio–gesture synchronization  
- **L1Div Score (↑)**: Higher → more gesture diversity  

**Recommendation**: MeanFlow typically offers the best overall trade-off, combining strong FGD and L1Div scores with the fastest inference.

---

## 10. Experimental Results & Notes

![Beat Results](beat-new.png)

The results table compares **1-speaker** and **all-speaker** setups. “RAG-Gesture” refers to [**Retrieving Semantics from the Deep: an RAG Solution for Gesture Synthesis**](https://arxiv.org/abs/2412.06786), accepted to CVPR 2025.  

- 1-speaker scores are reported for speaker ID 2 (“scott”) to stay aligned with prior SOTA methods.  
- The 1-speaker statistics are directly taken from the RAG-Gesture repository and differ from the numbers shown in the current paper.

### 10.1 Model Performance Details

- The paper’s reported metrics focus on a 1-speaker setting (speaker ID 2, “scott”) to be consistent with earlier work.  
- All released pretrained models (RVQ-VAEs, Diffusion, Shortcut, MeanFlow) are trained on a 1-speaker setup.  
- To enable all-speaker usage, update the configuration files to include all speaker IDs.  
- On **April 16, 2025**, the pretrained RVQ-VAEs and Shortcut models were updated to include all speakers.  
- No additional hyperparameter tuning was done for the all-speaker case; it uses the same settings as the 1-speaker configuration.

### 10.2 Design Choices

- No speaker embedding is used so that the model can generalize to unseen speakers.  
- Gesture type labels are not used in this version. This reflects realistic conditions where gesture types are typically unavailable for novel speakers and environments.  
- If better FGD scores are your main priority, you may experiment with including gesture type information.

### 10.3 Code Structure Notes

- **Current version**: Clean, unified repository with MeanFlow fully supported.  
- **Legacy code**: Available under the `old` branch for historical context.  
- **ICCV 2025**: The work has been accepted to ICCV 2025 — thanks to all co-authors.

---

## 11. Demo Details

The demo is built on top of the Shortcut model for fast, interactive gesture generation:

```bash
python demo.py -c configs/shortcut_rvqvae_128_hf.yaml
```

**Features**

- Web-based UI for user interaction  
- Real-time or near real-time gesture synthesis  
- Support for custom audio and textual input  
- Visualization of the generated gesture sequences  

---

## 12. Acknowledgments

We gratefully acknowledge the following projects, from which this repository borrows ideas and code:

- [SynTalker](https://github.com/RobinWitch/SynTalker/tree/main)  
- [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024)  
- [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture)  

Please consider exploring and citing these works as well.

---

## 13. Citation

If this codebase or the GestureLSM paper is helpful to your work, please consider citing:

```bibtex
@inproceedings{liu2025gesturelsmlatentshortcutbased,
  title={{GestureLSM: Latent Shortcut based Co-Speech Gesture Generation with Spatial-Temporal Modeling}},
  author={Pinxin Liu and Luchuan Song and Junhua Huang and Chenliang Xu},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  year={2025},
}
```
```