# Forensic Audio Authentication System

A production-ready deep learning system for detecting audio tampering,
splicing, and deepfake injection — built for legal and investigative use.

## Architecture

```
Input Audio → LFCC Features → CNN (spectral patterns) → BiLSTM (temporal) → Softmax → Verdict
```

## Quick Start

```bash
# 1. Clone and enter project
cd forensic_audio_auth

# 2. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add audio data
#    data/authentic/  → real speech (.wav/.flac)
#    data/deepfakes/  → ASVspoof2019 LA + WaveFake samples

# 5. Generate tampered clips (1000 by default)
python tampering/generate_dataset.py

# 6. Build dataset splits
python features/preprocess.py

# 7. Train
python model/train.py

# 8. Evaluate
python model/evaluate.py

# 9. Launch web app
python app/app.py
```

## Datasets

| Dataset | Label | Source |
|---------|-------|--------|
| TIMIT | Authentic (0) | LDC / OpenSLR |
| ASVspoof2019 LA | Tampered (1) | https://datashare.ed.ac.uk/handle/10283/3336 |
| WaveFake | Tampered (1) | https://github.com/RUB-SysSec/WaveFake |
| MUSAN (noise) | — | https://openslr.org/17/ |

## Metrics (targets)

| Metric | Target |
|--------|--------|
| EER | < 5% |
| AUC | > 0.95 |
| Precision@1%FAR | > 80% |

## Project Structure

```
forensic_audio_auth/
├── data/               Raw audio organized by class
├── features/           LFCC extraction + PyTorch Dataset
├── tampering/          Automated tampering simulation
├── model/              CNN-BiLSTM architecture + training + evaluation
├── xai/                Grad-CAM + SHAP explainability
├── app/                Gradio web interface + inference engine
├── notebooks/          EDA and result analysis
├── logs/               Training logs + TensorBoard runs
├── evaluation/         Saved plots (ROC, confusion matrix)
├── config.yaml         Central configuration
└── requirements.txt    Python dependencies
```

## TensorBoard

```bash
tensorboard --logdir logs/
```

## License
MIT
