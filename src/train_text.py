# Converted from 1_train_text.ipynb

# %% [markdown]
# # Text Emotion Training - Multi-Dataset
# 
# Trains RoBERTa on multiple text emotion datasets with unified labels.
# 
# **Datasets:**
# - simaanjali/emotion-analysis-based-on-text
# - nelgiriyewithana/emotions
# - GoEmotions (HuggingFace)
# 
# **Run in Google Colab with GPU!**

# %% [code]
# ============================================================
# STEP 1: Setup Kaggle API
# ============================================================
import os

# Create kaggle.json with your API key
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
# !pip install transformers datasets accelerate kaggle pandas scikit-learn numpy -q

# %% [code]
# ============================================================
# STEP 3: Download Datasets from Kaggle
# ============================================================
# !mkdir -p ./datasets/text

# Dataset 1: Emotion Analysis Based on Text
# !kaggle datasets download -d simaanjali/emotion-analysis-based-on-text -p ./datasets/text/simaanjali --unzip

# Dataset 2: Emotions Dataset
# !kaggle datasets download -d nelgiriyewithana/emotions -p ./datasets/text/emotions --unzip

print('\nDatasets downloaded!')

# %% [code]
# ============================================================
# STEP 4: Label Translation Map (CRITICAL!)
# ============================================================
# This unifies labels from all datasets into a common format

# Our unified 7 basic emotions
UNIFIED_LABELS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']

# Translation map for each dataset
LABEL_TRANSLATION = {
    # simaanjali/emotion-analysis-based-on-text
    'joy': 'happy',
    'happiness': 'happy',
    'love': 'happy',
    'sadness': 'sad',
    'grief': 'sad',
    'anger': 'angry',
    'rage': 'angry',
    'annoyance': 'angry',
    'fear': 'fear',
    'anxiety': 'fear',
    'nervousness': 'fear',
    'surprise': 'surprise',
    'shock': 'surprise',
    'disgust': 'disgust',
    'contempt': 'disgust',
    'neutral': 'neutral',
    'calm': 'neutral',
    
    # nelgiriyewithana/emotions
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    
    # GoEmotions (28 -> 7 mapping)
    'admiration': 'happy',
    'amusement': 'happy',
    'approval': 'happy',
    'caring': 'happy',
    'desire': 'happy',
    'excitement': 'happy',
    'gratitude': 'happy',
    'optimism': 'happy',
    'pride': 'happy',
    'relief': 'happy',
    'disappointment': 'sad',
    'embarrassment': 'sad',
    'remorse': 'sad',
    'confusion': 'neutral',
    'curiosity': 'neutral',
    'realization': 'surprise',
    'disapproval': 'angry',
}

def translate_label(label):
    """Translate any label to unified format."""
    label_lower = str(label).lower().strip()
    return LABEL_TRANSLATION.get(label_lower, label_lower)

# Convert to numeric labels
LABEL_TO_ID = dict((label, i) for i, label in enumerate(UNIFIED_LABELS))
ID_TO_LABEL = dict((i, label) for i, label in enumerate(UNIFIED_LABELS))

print('Unified Labels: {}'.format(UNIFIED_LABELS))
print('Label to ID: {}'.format(LABEL_TO_ID))

# %% [code]
# ============================================================
# STEP 5: Load and Merge Datasets
# ============================================================
import pandas as pd
import os
from datasets import load_dataset, Dataset, concatenate_datasets

all_texts = []
all_labels = []

# --- Dataset 1: simaanjali/emotion-analysis-based-on-text ---
try:
    for file in os.listdir('./datasets/text/simaanjali'):
        if file.endswith('.csv'):
            df = pd.read_csv('./datasets/text/simaanjali/{}'.format(file))
            # Find text and label columns (may vary)
            text_col = [c for c in df.columns if 'text' in c.lower() or 'content' in c.lower() or 'sentence' in c.lower()]
            label_col = [c for c in df.columns if 'label' in c.lower() or 'emotion' in c.lower() or 'class' in c.lower()]
            if text_col and label_col:
                texts = df[text_col[0]].tolist()
                labels = [translate_label(l) for l in df[label_col[0]].tolist()]
                all_texts.extend(texts)
                all_labels.extend(labels)
                print('Loaded {} samples from {}'.format(len(texts), file))
except Exception as e:
    print('Error loading simaanjali dataset: {}'.format(e))

# --- Dataset 2: nelgiriyewithana/emotions ---
try:
    for file in os.listdir('./datasets/text/emotions'):
        if file.endswith('.csv'):
            df = pd.read_csv('./datasets/text/emotions/{}'.format(file))
            text_col = [c for c in df.columns if 'text' in c.lower()]
            label_col = [c for c in df.columns if 'label' in c.lower() or 'emotion' in c.lower()]
            if text_col and label_col:
                texts = df[text_col[0]].tolist()
                labels = [translate_label(l) for l in df[label_col[0]].tolist()]
                all_texts.extend(texts)
                all_labels.extend(labels)
                print('Loaded {} samples from {}'.format(len(texts), file))
except Exception as e:
    print('Error loading emotions dataset: {}'.format(e))

# --- Dataset 3: GoEmotions from HuggingFace ---
try:
    go_emotions = load_dataset('go_emotions', 'simplified', split='train')
    GO_EMOTIONS_LABELS = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral']
    for item in go_emotions:
        all_texts.append(item['text'])
        # Use first label
        label_idx = item['labels'][0] if item['labels'] else 27
        original_label = GO_EMOTIONS_LABELS[label_idx]
        all_labels.append(translate_label(original_label))
    print('Loaded {} samples from GoEmotions'.format(len(go_emotions)))
except Exception as e:
    print('Error loading GoEmotions: {}'.format(e))

print('\\nTotal samples: {}'.format(len(all_texts)))
print('Label distribution: {}'.format(pd.Series(all_labels).value_counts().to_dict()))

# %% [code]
# ============================================================
# STEP 6: Convert to Numeric Labels and Create Dataset
# ============================================================

# Filter out invalid labels
valid_data = [(t, l) for t, l in zip(all_texts, all_labels) if l in LABEL_TO_ID]
texts_clean = [t for t, l in valid_data]
labels_clean = [LABEL_TO_ID[l] for t, l in valid_data]

print('Valid samples: {}'.format(len(texts_clean)))

# Create HuggingFace Dataset
from datasets import Dataset
dataset = Dataset.from_dict({'text': texts_clean, 'label': labels_clean})

# Train/val/test split
dataset = dataset.shuffle(seed=42)
train_test = dataset.train_test_split(test_size=0.2)
test_val = train_test['test'].train_test_split(test_size=0.5)

dataset_dict = {
    'train': train_test['train'],
    'validation': test_val['train'],
    'test': test_val['test']
}

print("Train: {}".format(len(dataset_dict['train'])))
print("Val: {}".format(len(dataset_dict['validation'])))
print("Test: {}".format(len(dataset_dict['test'])))

# %% [code]
# ============================================================
# STEP 7: Tokenize and Prepare Model
# ============================================================
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = 'roberta-base'
NUM_LABELS = len(UNIFIED_LABELS)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID
)

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128)

tokenized_train = dataset_dict['train'].map(tokenize, batched=True)
tokenized_val = dataset_dict['validation'].map(tokenize, batched=True)
tokenized_test = dataset_dict['test'].map(tokenize, batched=True)

print('Tokenization complete!')

# %% [code]
# ============================================================
# STEP 8: Training
# ============================================================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted')
    }

args = TrainingArguments(
    output_dir='./text_model_unified',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
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
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print('Starting training...')
trainer.train()

# %% [code]
# ============================================================
# STEP 9: Evaluate and Save
# ============================================================
results = trainer.evaluate(tokenized_test)
print("\\nTest Accuracy: {:.4f}".format(results['eval_accuracy']))
print("Test F1: {:.4f}".format(results['eval_f1']))

# Save to local models folder
SAVE_PATH = '../models/text_emotion_unified'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print('\\nModel saved to {}'.format(SAVE_PATH))

# %% [code]
# ============================================================
# STEP 10: Upload to HuggingFace (Optional)
# ============================================================
from huggingface_hub import login

# Uncomment and run to upload
login()  # Enter your HF token
model.push_to_hub('BasantAwad/text-emotion-detction')
tokenizer.push_to_hub('BasantAwad/text-emotion-detction')

# %% [code]
# ============================================================
# STEP 11: Quick Test
# ============================================================
from transformers import pipeline

classifier = pipeline('text-classification', model=SAVE_PATH, top_k=3)

test_texts = [
    "I'm so happy today!",
    "This makes me really angry",
    "I feel scared about tomorrow",
    "ewww that's gross",
    "wow I didn't expect that!"
]

for text in test_texts:
    result = classifier(text)
    print('\\n"{}"'.format(text))
    for r in result[0]:
        print("  {}: {:.3f}".format(r['label'], r['score']))

