# 🚀 Beginner's Guide: Running Multimodal Emotion AI on Google Colab

A complete step-by-step tutorial for running all three training notebooks.

---

## 📋 Prerequisites

- Google account (for Colab and Drive)
- Basic Python knowledge
- ~5GB Google Drive space

---

## Part 1: Setup Google Colab

### Step 1: Open Google Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account

### Step 2: Enable GPU (CRITICAL!)

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** under "Hardware accelerator"
3. Click **Save**

### Step 3: Mount Google Drive

Run this cell first in any notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

- Click the link that appears
- Allow access
- Copy the authorization code and paste it

---

## Part 2: Download Datasets

### Dataset 1: GoEmotions (Text)

**No download needed!** The notebook loads it directly from HuggingFace.

### Dataset 2: RAVDESS (Audio)

1. Go to [Kaggle RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
2. Click **Download** (you need a Kaggle account)
3. Upload the ZIP to Google Drive
4. Unzip in Colab:

```python
!unzip "/content/drive/MyDrive/ravdess.zip" -d "/content/drive/MyDrive/data/audio_data"
```

### Dataset 3: FER2013 (Face)

1. Go to [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Click **Download**
3. Upload to Google Drive
4. Unzip in Colab:

```python
!unzip "/content/drive/MyDrive/fer2013.zip" -d "/content/drive/MyDrive/data/face_data"
```

---

## Part 3: Upload Notebooks to Colab

### Method A: Upload from your computer

1. In Colab, click **File** → **Upload notebook**
2. Upload each `.ipynb` file one at a time

### Method B: Upload to Google Drive first

1. Upload `notebooks/` folder to Google Drive
2. In Colab: **File** → **Open notebook** → **Google Drive**
3. Navigate to your notebook

---

## Part 4: Run Training Notebooks

### 🗂️ Notebook 1: Text Emotion (RoBERTa)

**File:** `1_train_text.ipynb`
**Time:** ~30-45 minutes

#### Steps:

1. Open the notebook in Colab
2. Run each cell from top to bottom (Shift+Enter)
3. **Cell 1:** Installs libraries (wait for completion)
4. **Cell 2-4:** Loads dataset and tokenizer
5. **Cell 5-6:** Sets up training
6. **Cell 7:** Trains the model (this takes longest)
7. **Cell 8:** Saves model to `/content/drive/MyDrive/models/roberta_text/`

#### 💡 Tips:

- If you get "out of memory", reduce `per_device_train_batch_size` from 16 to 8
- Don't worry about warnings about "Some weights not initialized"

#### ⏩ Shortcut (Skip Training):

If you want to skip training, the notebook has a pre-trained model option:

```python
# Use this instead of training:
from transformers import pipeline
classifier = pipeline('text-classification', model='SamLowe/roberta-base-go_emotions')
```

---

### 🎵 Notebook 2: Audio Emotion (Wav2Vec)

**File:** `2_train_audio.ipynb`
**Time:** ~1-2 hours

#### Before Running:

Make sure RAVDESS data is in the right place:

```
/content/drive/MyDrive/data/audio_data/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-01-01-02-01-01.wav
│   └── ...
├── Actor_02/
└── ... (24 actors total)
```

#### Steps:

1. Open notebook
2. **Update DATA_PATH** in cell 4:

```python
DATA_PATH = '/content/drive/MyDrive/data/audio_data'
```

3. Run all cells
4. Model saves to `/content/drive/MyDrive/models/wav2vec_audio/`

#### ⚠️ Common Issues:

- **Out of Memory:** Reduce batch_size to 2
- **"No audio files found":** Check your folder structure matches above

---

### 😀 Notebook 3: Face Emotion (ResNet50)

**File:** `3_train_face.ipynb`
**Time:** ~1-2 hours

#### Before Running:

Make sure FER2013 data is structured as:

```
/content/drive/MyDrive/data/face_data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    └── ... (same folders)
```

#### Steps:

1. Open notebook
2. **Update DATA_PATH** in cell 2:

```python
DATA_PATH = '/content/drive/MyDrive/data/face_data'
```

3. Run all cells
4. Model saves to `/content/drive/MyDrive/models/resnet_face/`

---

## Part 5: Download Trained Models

After training all three notebooks:

1. In Colab, click the **folder icon** (📁) on the left
2. Navigate to `/content/drive/MyDrive/models/`
3. Right-click each folder → **Download**
4. Place downloaded folders in your local project's `/models/` directory

---

## Part 6: Run Inference Locally

### Step 1: Setup Local Environment

```bash
cd grad-project-emotion-ai
pip install -r requirements.txt
```

### Step 2: Test the System

```bash
# Demo mode (uses pre-trained HuggingFace models)
python main_api.py --mode demo

# Analyze a video file
python main_api.py --mode analyze --video path/to/video.mp4

# Start REST API
python main_api.py --mode api
```

---

## 🆘 Troubleshooting

### Problem: "CUDA out of memory"

**Solution:** Reduce batch size:

```python
per_device_train_batch_size=4  # or even 2
```

### Problem: "Runtime disconnected"

**Solution:**

- Keep the browser tab active
- Click "Reconnect" button
- For long training, use checkpoints to resume

### Problem: "No such file or directory"

**Solution:** Check paths! In Colab, paths should start with `/content/drive/MyDrive/`

### Problem: Model training is stuck

**Solution:** Check if GPU is enabled (Runtime → Change runtime type)

---

## ✅ Checklist

- [ ] GPU enabled in Colab
- [ ] Google Drive mounted
- [ ] RAVDESS uploaded and unzipped
- [ ] FER2013 uploaded and unzipped
- [ ] Notebook 1 (Text) completed
- [ ] Notebook 2 (Audio) completed
- [ ] Notebook 3 (Face) completed
- [ ] Models downloaded to local machine
- [ ] Local inference working

---

## 📚 Quick Reference

| Notebook      | Model    | Dataset           | Output Path            |
| ------------- | -------- | ----------------- | ---------------------- |
| 1_train_text  | RoBERTa  | GoEmotions (auto) | /models/roberta_text/  |
| 2_train_audio | Wav2Vec2 | RAVDESS           | /models/wav2vec_audio/ |
| 3_train_face  | ResNet50 | FER2013           | /models/resnet_face/   |

**Good luck with your grad project! 🎓**
