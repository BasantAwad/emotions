# Multimodal Emotion AI - Grad Project

A comprehensive multimodal emotion recognition system using three specialist models.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT                              │
└─────────────────┬───────────────┬───────────────┬───────────┘
                  │               │               │
                  ▼               ▼               ▼
         ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
         │   Specialist 1 │ │   Specialist 2 │ │   Specialist 3 │
         │   TEXT/RoBERTa │ │   AUDIO/Wav2Vec│ │   FACE/ResNet50│
         └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   ▼
                        ┌─────────────────┐
                        │ EMOTION FUSION  │
                        │    (Manager)    │
                        └────────┬────────┘
                                 ▼
                        ┌─────────────────┐
                        │  FINAL EMOTION  │
                        └─────────────────┘
```

## 📁 Project Structure

```
/grad-project-emotion-ai
├── /data/                   # Training data (download separately)
│   ├── /text_data/          # GoEmotions CSV files
│   ├── /audio_data/         # RAVDESS WAV files (Actor_XX folders)
│   └── /face_data/          # FER2013 images (train/ and test/)
│
├── /models/                 # Saved model weights after training
│   ├── /roberta_text/       # Fine-tuned RoBERTa
│   ├── /wav2vec_audio/      # Fine-tuned Wav2Vec 2.0
│   └── /resnet_face/        # Fine-tuned ResNet50
│
├── /notebooks/              # Google Colab training notebooks
│   ├── 1_train_text.ipynb   # Text emotion training
│   ├── 2_train_audio.ipynb  # Audio emotion training
│   └── 3_train_face.ipynb   # Face emotion training
│
├── /src/                    # Inference code (predictors)
│   ├── text_predictor.py    # TextEmotionAnalyzer class
│   ├── audio_predictor.py   # AudioEmotionAnalyzer class
│   ├── face_predictor.py    # FaceEmotionAnalyzer class
│   └── utils.py             # Shared utilities
│
├── main_api.py              # Integration manager + REST API
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

- **Text**: [GoEmotions on HuggingFace](https://huggingface.co/datasets/go_emotions)
- **Audio**: [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Face**: [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

### 3. Train Models (in Google Colab with GPU)

Upload notebooks to Colab and run:

1. `1_train_text.ipynb` - ~30 min
2. `2_train_audio.ipynb` - ~1-2 hours
3. `3_train_face.ipynb` - ~1-2 hours

### 4. Run Inference

```bash
# Demo mode
python main_api.py --mode demo

# Analyze a video
python main_api.py --mode analyze --video path/to/video.mp4

# Start REST API server
python main_api.py --mode api --port 8000
```

## 📊 Models

| Specialist | Model        | Dataset    | Emotions   |
| ---------- | ------------ | ---------- | ---------- |
| Text       | RoBERTa-base | GoEmotions | 28 classes |
| Audio      | Wav2Vec 2.0  | RAVDESS    | 8 classes  |
| Face       | ResNet50     | FER2013    | 7 classes  |

## 🔗 Pre-trained Model Shortcuts

If you skip training, use these HuggingFace models:

- **Text**: `SamLowe/roberta-base-go_emotions`
- **Audio**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`

## 📝 API Endpoints

When running in API mode:

- `POST /analyze/text` - Analyze text emotion
- `POST /analyze/audio` - Analyze audio file
- `POST /analyze/face` - Analyze face image
- `POST /analyze/video` - Full multimodal analysis

## 🎯 Emotion Fusion Logic

The system combines predictions using:

- **Unanimous**: All 3 models agree → High confidence
- **Majority**: 2/3 agree → Use majority vote
- **Conflict**: All different → Use weighted confidence
- **Special cases**: Detect sarcasm, masked emotions
