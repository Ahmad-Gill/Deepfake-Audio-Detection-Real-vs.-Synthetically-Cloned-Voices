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

## 🤖 AI Skills & Development Guides

This repository includes **AI-powered development skills** in the `skills/` directory to accelerate your work:

### Available Skills
- **`implementation-guide/`** - Complete coding, training, and debugging guidance
- **`midterm-report-writer/`** - Academic report writing assistance  
- **`paper-reader/`** - Research paper analysis and summarization

### How to Use Skills
1. **For Implementation Work**: Ask anything about coding, training, debugging
   - *"Help me implement RawNet2"*
   - *"My training isn't converging"*
   - *"How do I compute EER?"*

2. **For Report Writing**: Get help with academic writing
   - *"Write my methodology section"*
   - *"Help me analyze these results"*

3. **For Research**: Get paper summaries and insights
   - *"Summarize the AASIST paper"*
   - *"What are the key findings from RawBoost?"*

### Quick Skill Reference
```bash
# When you need help, just ask naturally:
"Set up my environment"           → Environment setup guide
"Debug my training"              → Debugging assistance  
"Run experiments"                → Experiment planning
"Write evaluation section"       → Report writing help
```