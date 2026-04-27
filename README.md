# Deepfake Audio Detection: Real vs. Synthetically Cloned Voices

This project focuses on detecting whether an audio clip is:
- Genuine human speech
- AI-generated / cloned speech (spoofed)

Using the ASVspoof 2019 Logical Access dataset, we build robust binary classification models capable of generalizing to unseen spoofing attacks.

## Objectives
- Build deep learning models for audio anti-spoofing
- Extract and analyze audio features (MFCC, Spectrograms, CQCC)
- Improve robustness with augmentation techniques
- Evaluate generalization on unseen attacks
- Study failure cases and security implications of synthetic voice detection

## Dataset
ASVspoof 2019 Logical Access (121k+ utterances, 107 speakers, 19 spoofing systems)

Task:
Binary Classification:
- Bonafide (Real)
- Spoof (Synthetic)

Potential Models:
- CNN
- CRNN
- ResNet
- LSTM
- Transformer-based audio models