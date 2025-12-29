# Converted from 2_train_audio.ipynb

# %% [markdown]
# # Audio Emotion Training - Multi-Dataset
# 
# Trains Wav2Vec 2.0 on multiple audio emotion datasets with unified labels.
# 
# **Datasets:**
# - RAVDESS
# - dmitrybabko/speech-emotion-recognition-en
# - ejlok1/toronto-emotional-speech-set-tess
# 
# **Run in Google Colab with GPU!**

# %% [code]
# ============================================================
# STEP 1: Setup Kaggle API
# ============================================================
import os

kaggle_json = '''{
    "username": "basantawad",
    "key": "73699caea5f0322acca5bc42516c5998"
}'''

kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
with open(kaggle_file, 'w') as f:
    f.write(kaggle_json)
os.chmod(kaggle_file, 0o600)
print('Kaggle API Connected!')

# %% [code]
# ============================================================
# STEP 2: Install Libraries
# ============================================================
# !pip install transformers datasets accelerate kaggle librosa soundfile pandas scikit-learn numpy -q

# %% [code]
# ============================================================
# STEP 3: Download Datasets from Kaggle
# ============================================================
# !mkdir -p ./datasets/audio

# Dataset 1: RAVDESS
# !kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p ./datasets/audio/ravdess --unzip

# Dataset 2: Speech Emotion Recognition EN
# !kaggle datasets download -d dmitrybabko/speech-emotion-recognition-en -p ./datasets/audio/ser_en --unzip

# Dataset 3: TESS (Toronto Emotional Speech Set)
# !kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess -p ./datasets/audio/tess --unzip

print('\nDatasets downloaded!')

# %% [code]
# ============================================================
# STEP 4: Label Translation Map (CRITICAL!)
# ============================================================
UNIFIED_LABELS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']

# Translation map for audio datasets
LABEL_TRANSLATION = {
    # RAVDESS labels (from filename codes)
    'neutral': 'neutral',
    'calm': 'neutral',
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'fearful': 'fear',
    'fear': 'fear',
    'disgust': 'disgust',
    'surprised': 'surprise',
    'surprise': 'surprise',
    
    # Speech Emotion Recognition EN
    'happiness': 'happy',
    'sadness': 'sad',
    'anger': 'angry',
    
    # TESS labels
    'ps': 'surprise',  # pleasant surprise
    'pleasant_surprise': 'surprise',
    'pleasantlytsurprised': 'surprise',
}

# RAVDESS emotion code mapping
RAVDESS_CODES = {
    '01': 'neutral',
    '02': 'neutral',  # calm -> neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

def translate_label(label):
    label_lower = str(label).lower().strip()
    return LABEL_TRANSLATION.get(label_lower, label_lower)

LABEL_TO_ID = {label: i for i, label in enumerate(UNIFIED_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(UNIFIED_LABELS)}

print(f'Unified Labels: {UNIFIED_LABELS}')

# %% [code]
# ============================================================
# STEP 5: Load and Process Audio Files
# ============================================================
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

SAMPLE_RATE = 16000
audio_files = []
audio_labels = []

# --- RAVDESS ---
print('Loading RAVDESS...')
ravdess_path = Path('./datasets/audio/ravdess')
for wav_file in ravdess_path.rglob('*.wav'):
    filename = wav_file.stem
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_code = parts[2]
        if emotion_code in RAVDESS_CODES:
            audio_files.append(str(wav_file))
            audio_labels.append(RAVDESS_CODES[emotion_code])
print(f'RAVDESS: {len([l for l in audio_labels])} files')

# --- TESS ---
print('Loading TESS...')
tess_path = Path('./datasets/audio/tess')
tess_count = 0
for wav_file in tess_path.rglob('*.wav'):
    filename = wav_file.stem.lower()
    # TESS format: OAF_angry_xxx or YAF_happy_xxx
    parts = filename.split('_')
    for part in parts:
        translated = translate_label(part)
        if translated in UNIFIED_LABELS:
            audio_files.append(str(wav_file))
            audio_labels.append(translated)
            tess_count += 1
            break
print(f'TESS: {tess_count} files')

# --- Speech Emotion Recognition EN ---
print('Loading SER-EN...')
ser_path = Path('./datasets/audio/ser_en')
ser_count = 0
for wav_file in ser_path.rglob('*.wav'):
    # Try to extract emotion from folder name or filename
    parent_name = wav_file.parent.name.lower()
    translated = translate_label(parent_name)
    if translated in UNIFIED_LABELS:
        audio_files.append(str(wav_file))
        audio_labels.append(translated)
        ser_count += 1
print(f'SER-EN: {ser_count} files')

print(f'\nTotal audio files: {len(audio_files)}')
print(f'Label distribution: {dict(zip(*np.unique(audio_labels, return_counts=True)))}')

# %% [code]
# ============================================================
# STEP 6: Create Dataset and Split
# ============================================================
from sklearn.model_selection import train_test_split

# Convert labels to IDs
label_ids = [LABEL_TO_ID[l] for l in audio_labels]

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    audio_files, label_ids, test_size=0.2, stratify=label_ids, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

# %% [code]
# ============================================================
# STEP 7: Create PyTorch Dataset
# ============================================================
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

class AudioEmotionDataset(Dataset):
    def __init__(self, files, labels, feature_extractor, max_length=16000*5):
        self.files = files
        self.labels = labels
        self.fe = feature_extractor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load audio
        wav, _ = librosa.load(self.files[idx], sr=SAMPLE_RATE)
        
        # Pad or truncate
        if len(wav) > self.max_length:
            wav = wav[:self.max_length]
        elif len(wav) < self.max_length:
            wav = np.pad(wav, (0, self.max_length - len(wav)))
        
        # Process
        inputs = self.fe(wav, sampling_rate=SAMPLE_RATE, return_tensors='pt', padding=False)
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

# Load feature extractor and model
MODEL_NAME = 'facebook/wav2vec2-base'
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(UNIFIED_LABELS),
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID
)

# Create datasets
train_dataset = AudioEmotionDataset(X_train, y_train, feature_extractor)
val_dataset = AudioEmotionDataset(X_val, y_val, feature_extractor)
test_dataset = AudioEmotionDataset(X_test, y_test, feature_extractor)

print('Datasets created!')

# %% [code]
# ============================================================
# STEP 8: Training
# ============================================================
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted')
    }

args = TrainingArguments(
    output_dir='./audio_model_unified',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=500,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    fp16=torch.cuda.is_available(),
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print('Starting training (this takes 1-2 hours)...')
trainer.train()

# %% [code]
# ============================================================
# STEP 9: Evaluate and Save
# ============================================================
results = trainer.evaluate(test_dataset)
print(f"\nTest Accuracy: {results['eval_accuracy']:.4f}")
print(f"Test F1: {results['eval_f1']:.4f}")

# Save
SAVE_PATH = '../models/audio_emotion_unified'
os.makedirs(SAVE_PATH, exist_ok=True)
trainer.save_model(SAVE_PATH)
feature_extractor.save_pretrained(SAVE_PATH)
print(f'\nModel saved to {SAVE_PATH}')

# %% [code]
# ============================================================
# STEP 10: Upload to HuggingFace (Optional)
# ============================================================
 from huggingface_hub import login
 login()
 model.push_to_hub('BasantAwad/speech_emotion')
 feature_extractor.push_to_hub('BasantAwad/speech_emotion')

