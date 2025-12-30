# Converted from 3_train_face.ipynb

# %% [markdown]
# # Face Emotion Training - Multi-Dataset
# 
# Trains Vision Transformer (ViT) on multiple face emotion datasets with unified labels.
# 
# **Datasets:**
# - FER2013 (msambare/fer2013)
# - fahadullaha/facial-emotion-recognition-dataset
# - sujaykapadnis/emotion-recognition-dataset
# - ananthu017/emotion-detection-fer
# 
# **Run in Google Colab with GPU!**

# %% [code]
# ============================================================
# STEP 1: Setup Kaggle API
# ============================================================
import os

# Select GPU 4 (least busy based on nvidia-smi)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

kaggle_json = '''{
    "username": "basantawad",
    "key": "73699caea5f0322acca5bc42516c5998"
}'''

kaggle_dir = os.path.expanduser('~/.kaggle')
if not os.path.exists(kaggle_dir):
    os.makedirs(kaggle_dir)
kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
with open(kaggle_file, 'w') as f:
    f.write(kaggle_json)
os.chmod(kaggle_file, 0o600)
print('Kaggle API Connected!')

# %% [code]
# ============================================================
# STEP 2: Install Libraries
# ============================================================
# !pip install transformers datasets accelerate kaggle Pillow tqdm scikit-learn pandas numpy -q

# %% [code]
# ============================================================
# STEP 3: Download Datasets from Kaggle
# ============================================================
import subprocess

# Create directories
if not os.path.exists('./datasets/face'):
    os.makedirs('./datasets/face')

# Dataset 1: FER2013 (main dataset)
if not os.path.exists('./datasets/face/fer2013'):
    print('Downloading FER2013...')
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'msambare/fer2013', '-p', './datasets/face/fer2013', '--unzip'])

# Dataset 2: Facial Emotion Recognition (optional)
if not os.path.exists('./datasets/face/fer_dataset'):
    print('Downloading Facial Emotion Recognition dataset...')
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'fahadullaha/facial-emotion-recognition-dataset', '-p', './datasets/face/fer_dataset', '--unzip'])

# Dataset 3: Emotion Recognition Dataset (optional)
if not os.path.exists('./datasets/face/emotion_rec'):
    print('Downloading Emotion Recognition dataset...')
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'sujaykapadnis/emotion-recognition-dataset', '-p', './datasets/face/emotion_rec', '--unzip'])

# Dataset 4: Emotion Detection FER (optional)
if not os.path.exists('./datasets/face/emotion_det'):
    print('Downloading Emotion Detection FER...')
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'ananthu017/emotion-detection-fer', '-p', './datasets/face/emotion_det', '--unzip'])

print('\\nDatasets downloaded!')

# %% [code]
# ============================================================
# STEP 4: Label Translation Map (CRITICAL!)
# ============================================================
UNIFIED_LABELS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']

LABEL_TRANSLATION = {
    # FER2013 standard labels
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprise',
    'neutral': 'neutral',
    
    # Variations
    'happiness': 'happy',
    'sadness': 'sad',
    'anger': 'angry',
    'fearful': 'fear',
    'surprised': 'surprise',
    'disgusted': 'disgust',
    
    # Additional labels some datasets might have
    'contempt': 'disgust',
    'joy': 'happy',
}

def translate_label(label):
    label_lower = str(label).lower().strip()
    return LABEL_TRANSLATION.get(label_lower, label_lower)

LABEL_TO_ID = dict((label, i) for i, label in enumerate(UNIFIED_LABELS))
ID_TO_LABEL = dict((i, label) for i, label in enumerate(UNIFIED_LABELS))

print('Unified Labels: {}'.format(UNIFIED_LABELS))

# %% [code]
# ============================================================
# STEP 5: Load Images from All Datasets
# ============================================================
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

image_paths = []
image_labels = []

def process_folder(base_path, dataset_name):
    """Process images organized in emotion folders."""
    count = 0
    for emotion_folder in Path(base_path).iterdir():
        if emotion_folder.is_dir():
            folder_name = emotion_folder.name.lower()
            translated = translate_label(folder_name)
            if translated in UNIFIED_LABELS:
                for img_file in emotion_folder.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_paths.append(str(img_file))
                        image_labels.append(translated)
                        count += 1
    print('{}: {} images'.format(dataset_name, count))

# FER2013
print('Loading FER2013...')
if Path('./datasets/face/fer2013/train').exists():
    process_folder('./datasets/face/fer2013/train', 'FER2013-train')
if Path('./datasets/face/fer2013/test').exists():
    process_folder('./datasets/face/fer2013/test', 'FER2013-test')

# Other datasets - search recursively for emotion folders
other_paths = [
    './datasets/face/fer_dataset',
    './datasets/face/emotion_rec',
    './datasets/face/emotion_det'
]

for base in other_paths:
    if Path(base).exists():
        # Try direct processing
        process_folder(base, base.split('/')[-1])
        # Also try train/test subfolders
        for sub in ['train', 'test', 'Train', 'Test', 'training', 'testing']:
            sub_path = Path(base) / sub
            if sub_path.exists():
                process_folder(str(sub_path), '{}-{}'.format(base.split("/")[-1], sub))

print('\nTotal images: {}'.format(len(image_paths)))
print('Label distribution: {}'.format(dict(zip(*np.unique(image_labels, return_counts=True)))))

# %% [code]
# ============================================================
# STEP 6: Create Dataset and Split
# ============================================================
from sklearn.model_selection import train_test_split

label_ids = [LABEL_TO_ID[l] for l in image_labels]

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, label_ids, test_size=0.2, stratify=label_ids, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print('Train: {}, Val: {}, Test: {}'.format(len(X_train), len(X_val), len(X_test)))

# %% [code]
# ============================================================
# STEP 7: Create HuggingFace Dataset
# ============================================================
from datasets import Dataset, Features, ClassLabel, Image as HFImage
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Create datasets
train_data = Dataset.from_dict({'image': X_train, 'label': y_train}).cast_column('image', HFImage())
val_data = Dataset.from_dict({'image': X_val, 'label': y_val}).cast_column('image', HFImage())
test_data = Dataset.from_dict({'image': X_test, 'label': y_test}).cast_column('image', HFImage())

# Load image processor and model (ViT for better accuracy)
MODEL_NAME = 'google/vit-base-patch16-224'

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(UNIFIED_LABELS),
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
    ignore_mismatched_sizes=True
)

print('Model and processor loaded!')

# %% [code]
# ============================================================
# STEP 8: Preprocessing
# ============================================================
def preprocess(examples):
    images = []
    for img in examples['image']:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
    inputs = processor(images=images, return_tensors='pt')
    inputs['labels'] = examples['label']
    return inputs

train_data = train_data.map(preprocess, batched=True, batch_size=32, remove_columns=['image'])
val_data = val_data.map(preprocess, batched=True, batch_size=32, remove_columns=['image'])
test_data = test_data.map(preprocess, batched=True, batch_size=32, remove_columns=['image'])

train_data.set_format('torch')
val_data.set_format('torch')
test_data.set_format('torch')

print('Preprocessing complete!')

# %% [code]
# ============================================================
# STEP 9: Training
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
    output_dir='./face_model_unified',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
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
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

print('Starting training (this takes 1-2 hours)...')
trainer.train()

# %% [code]
# ============================================================
# STEP 10: Evaluate and Save
# ============================================================
results = trainer.evaluate(test_data)
print("\nTest Accuracy: {:.4f}".format(results['eval_accuracy']))
print("Test F1: {:.4f}".format(results['eval_f1']))

# Save
SAVE_PATH = '../models/face_emotion_unified'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
trainer.save_model(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)
print('\nModel saved to {}'.format(SAVE_PATH))

# %% [code]
# ============================================================
# STEP 11: Upload to HuggingFace (Optional)
# ============================================================
from huggingface_hub import login
login()
model.push_to_hub('BasantAwad/facial-emotion')
processor.push_to_hub('BasantAwad/facial-emotion')


