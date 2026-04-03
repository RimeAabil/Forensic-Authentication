# 🎙️ Forensic Audio Authentication — Deep Learning

> Système de détection de manipulation audio basé sur le deep learning, capable d'identifier les enregistrements **authentiques**, **splicés**, et **deepfakes vocaux** avec une explicabilité judiciaire intégrée (Grad-CAM).

---

## 📌 Vue d'ensemble

Ce projet implémente un pipeline complet d'**authentification forensique audio** à destination des experts judiciaires et chercheurs en sécurité. Il couvre la simulation de splices, l'extraction de features LFCC, l'entraînement d'un modèle CNN-BiLSTM, et l'interprétabilité des décisions via Grad-CAM — le tout accessible via une interface Gradio.

| Classe détectée | Description |
|---|---|
| `0 — Authentique` | Enregistrement original non modifié |
| `1 — Manipulé` | Splice simulé, deepfake vocoder (WaveFake), ou TTS (ASVspoof) |

---

## 🏗️ Architecture du pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  DONNÉES BRUTES                                                       │
│  LJSpeech (13 100 clips) + WaveFake (6 vocoders) + ASVspoof2019 LA  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  NOTEBOOK 1 — PREPROCESSING                                          │
│  • Rééchantillonnage à 16 kHz                                        │
│  • Normalisation amplitude (−20 dBFS) et durée (4 s fixe)           │
│  • Simulation de 1 000 splices avec crossfade 20 ms                  │
│  • Construction de master_labels.csv (split 80/10/10)               │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  NOTEBOOK 2 — EXTRACTION LFCC                                        │
│  • LFCC (20 coefficients, N_FFT=400, Hop=160)                        │
│  • Normalisation CMVN par clip                                       │
│  • Sauvegarde en .npy — shape (20, 400) par clip                    │
│  • master_labels_with_features.csv                                   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  NOTEBOOK 3 — ENTRAÎNEMENT CNN-BiLSTM                                │
│                                                                      │
│  Input (batch, 1, 20, 400)                                           │
│       ↓                                                              │
│  CNN  : Conv2D×3 [32→64→128] + BatchNorm + MaxPool                  │
│       ↓                                                              │
│  Bi-LSTM : 2 couches, 256 hidden × 2 directions = 512               │
│       ↓                                                              │
│  Classifier : Linear(512→128→2)                                      │
│       ↓                                                              │
│  Output : [score_authentique, score_manipulé]                        │
│                                                                      │
│  • WeightedRandomSampler (déséquilibre 1:9)                          │
│  • Early stopping (patience=5), Grad Clipping                       │
│  • Export ONNX pour déploiement                                      │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  NOTEBOOK 4 — EXPLICABILITÉ (XAI)                                    │
│  • Grad-CAM sur la dernière couche CNN → heatmap temporelle          │
│  • Integrated Gradients (Captum) → importance des coefficients LFCC │
│  • Visualisations admissibles en tribunal                            │
│  • Validation : Grad-CAM pointe sur le frame de splice connu        │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  NOTEBOOK 5 — INTERFACE GRADIO                                       │
│  • Upload audio (.wav / .flac / .mp3 / .ogg)                        │
│  • Pipeline temps réel : audio → LFCC → modèle → verdict            │
│  • Affichage probabilité + heatmap Grad-CAM + rapport                │
│  • Inférence via PyTorch ou ONNX Runtime                            │
│  • Lien public Gradio (72h)                                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Prérequis

| Composant | Version minimale |
|---|---|
| Python | 3.10+ |
| CUDA | 11.8+ (recommandé) |
| GPU VRAM | 8 GB minimum (T4 Kaggle suffisant) |
| RAM | 16 GB minimum |
| Espace disque | ~50 GB (datasets + features) |

### Datasets requis

| Dataset | Source | Description |
|---|---|---|
| **LJSpeech-1.1** | [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech) | 13 100 clips vocaux authentiques |
| **WaveFake** | [Kaggle WaveFake](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfakes-voice-conversion) | Deepfakes par 6 vocoders différents |
| **ASVspoof2019 LA** | [ASVspoof Challenge](https://datashare.ed.ac.uk/handle/10283/3336) | Anti-spoofing, accès sur demande |

---

## 🚀 Installation & Configuration

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/forensic-audio-auth.git
cd forensic-audio-auth
```

### 2. Créer l'environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install \
    numpy==1.26.4 \
    pandas==2.2.2 \
    matplotlib \
    scikit-learn \
    tqdm \
    speechbrain \
    soundfile \
    captum \
    gradio \
    onnxruntime \
    Pillow
```

### 4. Structure des répertoires attendue

```
forensic-audio-auth/
├── notebooks/
│   ├── 01-preprocessing.ipynb
│   ├── 02-LFCC-extraction.ipynb
│   ├── 03-model-training.ipynb
│   ├── 04-xai-gradcam.ipynb
│   └── 05-gradio-app.ipynb
├── data/
│   ├── LJSpeech-1.1/wavs/          ← .wav LJSpeech
│   ├── generated_audio/            ← WaveFake par vocoder
│   └── LA/                         ← ASVspoof2019 LA
├── outputs/
│   ├── ljspeech_16k/               ← généré par NB1
│   ├── spliced/                    ← généré par NB1
│   ├── splits/
│   │   └── master_labels.csv       ← généré par NB1
│   ├── lfcc_features/              ← généré par NB2
│   ├── checkpoints/
│   │   └── best_model.pth          ← généré par NB3
│   └── models/
│       └── forensic_audio.onnx     ← généré par NB3
└── README.md
```

---

## 📖 Guide d'utilisation

Les notebooks doivent être **exécutés dans l'ordre**. Chaque notebook consomme les sorties du précédent.

### Étape 1 — Prétraitement audio

```bash
jupyter notebook notebooks/01-preprocessing.ipynb
```

**Produit :**
- `outputs/ljspeech_16k/` — clips rééchantillonnés à 16 kHz
- `outputs/spliced/` — 1 000 clips splicés simulés
- `outputs/splits/master_labels.csv` — index unifié avec splits 80/10/10

**Durée estimée :** 30–45 min sur GPU

---

### Étape 2 — Extraction des features LFCC

```bash
jupyter notebook notebooks/02-LFCC-extraction.ipynb
```

**Produit :**
- `outputs/lfcc_features/*.npy` — matrices LFCC (20, 400) par clip
- `outputs/master_labels_with_features.csv` — CSV enrichi avec chemins .npy

**Durée estimée :** 45–60 min sur GPU

---

### Étape 3 — Entraînement du modèle CNN-BiLSTM

```bash
jupyter notebook notebooks/03-model-training.ipynb
```

**Produit :**
- `outputs/checkpoints/best_model.pth` — meilleur checkpoint (val_loss)
- `outputs/models/forensic_audio.onnx` — modèle exporté pour déploiement
- `outputs/results/metrics.csv` — EER, FAR@1%FRR, AUC-ROC
- `outputs/training_curves.png` / `confusion_matrix.png`

**Durée estimée :** 1h30–2h sur GPU

---

### Étape 4 — Explicabilité Grad-CAM & SHAP

```bash
jupyter notebook notebooks/04-xai-gradcam.ipynb
```

**Produit :**
- `outputs/xai/gradcam_*.png` — heatmaps pour chaque clip analysé
- `outputs/xai/shap_*.png` — importance des coefficients LFCC
- `outputs/xai/courtroom_*.png` — visualisations prêtes pour rapport judiciaire
- `outputs/xai/splice_validation.png` — validation de localisation sur splice connu

**Durée estimée :** 30–45 min sur GPU

---

### Étape 5 — Interface Gradio

```bash
jupyter notebook notebooks/05-gradio-app.ipynb
```

Puis dans la cellule de lancement :

```python
# Lance l'interface avec lien public (valable 72h)
demo.launch(share=True)
```

L'interface accepte les formats `.wav`, `.flac`, `.mp3`, `.ogg` et retourne :
- Verdict (Authentique / Manipulé)
- Probabilité de manipulation (%)
- Heatmap Grad-CAM superposée sur le spectrogramme LFCC
- Rapport textuel exportable

---

## 🔧 Configuration

Les paramètres globaux sont définis en tête de chaque notebook. Les variables critiques à adapter selon votre environnement :

### Chemins des datasets

```python
# Dans 01-preprocessing.ipynb et 02-LFCC-extraction.ipynb
LJSPEECH_DIR  = Path('/chemin/vers/LJSpeech-1.1/wavs')
WAVEFAKE_DIR  = Path('/chemin/vers/generated_audio')
ASVSPOOF_DIR  = Path('/chemin/vers/LA')
OUTPUT_DIR    = Path('/chemin/vers/outputs')
```

### Paramètres audio

```python
SAMPLE_RATE    = 16000    # Fréquence d'échantillonnage cible (Hz)
CLIP_DURATION  = 4        # Durée fixe des clips (secondes)
N_LFCC         = 20       # Nombre de coefficients LFCC
N_FFT          = 400      # Taille de la fenêtre FFT (25 ms à 16 kHz)
HOP_LENGTH     = 160      # Pas de la fenêtre (10 ms à 16 kHz)
N_FILTER       = 128      # Nombre de filtres de la banque linéaire
N_SPLICES      = 1000     # Nombre de clips splicés à simuler
```

### Hyperparamètres d'entraînement

```python
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
N_EPOCHS       = 20
EARLY_STOPPING = 5        # Patience en nombre d'epochs
GRAD_CLIP      = 1.0
```

### Exécution sur Kaggle

Si vous exécutez sur Kaggle, adaptez les chemins de base :

```python
BASE     = Path('/kaggle/input/<votre-username>/<nom-dataset>')
OUTPUT_DIR = Path('/kaggle/working')
```

Activer le GPU dans **Settings → Accelerator → GPU T4 x2** avant d'exécuter.

---

## 📊 Métriques cibles

| Métrique | Objectif |
|---|---|
| EER (Equal Error Rate) | < 5% |
| AUC-ROC | > 0.95 |
| FAR @ 1% FRR | < 3% |
| Accuracy test set | > 95% |

---

## 👥 Contributeurs

| Nom | Rôle |
|---|---|
| **[Nom 1]** | Modélisation deep learning, entraînement, XAI |
| **[Nom 2]** | Prétraitement audio, simulation splices, interface Gradio |

---

## 📄 Licence

Ce projet est développé dans un cadre académique. Les datasets utilisés (LJSpeech, WaveFake, ASVspoof2019) sont soumis à leurs licences respectives — consulter les sources originales avant tout usage commercial.
