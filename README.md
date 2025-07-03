# BERT-LightRFFI: Large Language Model Enabled Lightweight RFFI for 6G Edge Intelligence

## 📖 概述
本仓库用于介绍在论文 **《Let RFF do the talking: large language model enabled lightweight RFFI for 6G edge intelligence》** 中所提出的BERT-LightRFFI框架，此论文已公开发表在*SCIENCE CHINA Information Sciences* (2025)。

The framework addresses security challenges in 6G edge IoT networks by:
1. Leveraging large language models (BERT) for radio frequency fingerprint (RFF) extraction
2. Using knowledge distillation to compress models for edge deployment
3. Achieving 97.52% accuracy under multipath fading and Doppler shift conditions
4. Reducing model size by 96.3% and computation by 89.2% compared to baselines

![BERT-LightRFFI Framework](https://via.placeholder.com/600x300?text=Framework+Diagram )

## 🚀 Key Features
- **LLM-enhanced RFFI**: First integration of BERT with radio frequency fingerprint identification
- **Lightweight Deployment**: <100KB model size suitable for edge IoT devices
- **Channel Robustness**: Maintains high accuracy under multipath fading (30dB SNR) and Doppler shift (20Hz)
- **Few-shot Learning**: Requires minimal labeled data (20% of dataset) for fine-tuning
- **Cross-modal Training**: Uses both wired and wireless signals for feature extraction

## ⚙️ Environment Setup
```bash
conda create -n rffi python=3.8
conda activate rffi
pip install -r requirements.txt
