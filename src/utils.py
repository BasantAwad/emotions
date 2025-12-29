"""
Utility Functions for Multimodal Emotion AI
Shared helper functions for file loading, video processing, and data handling.
"""

import os
import cv2
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import tempfile


# ============================================================================
# Constants
# ============================================================================
EMOTION_LABELS_FACE = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_LABELS_AUDIO = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SAMPLE_RATE = 16000  # Required for Wav2Vec 2.0


# ============================================================================
# Video Processing Functions
# ============================================================================
def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from a video file.
    
    Args:
        video_path: Path to the input video file.
        output_path: Optional path for the output audio file.
                     If None, creates a temp file.
    
    Returns:
        Path to the extracted audio file (WAV format).
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.wav')
    
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path, fps=SAMPLE_RATE, verbose=False, logger=None)
    video.close()
    
    return output_path


def extract_frames_from_video(
    video_path: str,
    interval_seconds: float = 1.0,
    output_dir: Optional[str] = None
) -> List[np.ndarray]:
    """
    Extract frames from a video at specified intervals.
    
    Args:
        video_path: Path to the input video file.
        interval_seconds: Time interval between frame captures.
        output_dir: Optional directory to save frames as images.
    
    Returns:
        List of frames as numpy arrays (BGR format).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    cap.release()
    return frames


def split_video(video_path: str) -> Tuple[str, List[np.ndarray]]:
    """
    Split a video into its audio track and frames.
    
    Args:
        video_path: Path to the input video file.
    
    Returns:
        Tuple of (audio_path, list_of_frames)
    """
    audio_path = extract_audio_from_video(video_path)
    frames = extract_frames_from_video(video_path)
    return audio_path, frames


# ============================================================================
# Audio Processing Functions
# ============================================================================
def load_audio(audio_path: str, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load and resample audio file to target sample rate.
    
    Args:
        audio_path: Path to the audio file.
        target_sr: Target sample rate (default: 16000 for Wav2Vec).
    
    Returns:
        Tuple of (audio_waveform, sample_rate)
    """
    waveform, sr = librosa.load(audio_path, sr=target_sr)
    return waveform, sr


def preprocess_audio_for_wav2vec(audio_path: str) -> np.ndarray:
    """
    Preprocess audio file for Wav2Vec 2.0 model.
    
    Args:
        audio_path: Path to the audio file.
    
    Returns:
        Audio waveform as numpy array, resampled to 16kHz.
    """
    waveform, _ = load_audio(audio_path, target_sr=SAMPLE_RATE)
    
    # Normalize audio
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    
    return waveform


# ============================================================================
# Image Processing Functions
# ============================================================================
def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file.
    
    Returns:
        Image as numpy array (BGR format).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def preprocess_image_for_resnet(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess image for ResNet50 model.
    Handles grayscale to RGB conversion and resizing.
    
    Args:
        image: Input image as numpy array.
        target_size: Target size (height, width) for the model.
    
    Returns:
        Preprocessed image ready for ResNet50.
    """
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        # Grayscale image - stack 3 times to create RGB
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1:
        # Single channel image
        image = np.concatenate([image, image, image], axis=-1)
    elif image.shape[-1] == 4:
        # BGRA to BGR
        image = image[:, :, :3]
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    image = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    return image


def detect_and_crop_face(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect and crop face from an image using OpenCV's Haar Cascade.
    
    Args:
        image: Input image as numpy array.
    
    Returns:
        Cropped face image or None if no face detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    # Get the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Add some padding
    padding = int(0.1 * max(w, h))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    return image[y:y+h, x:x+w]


# ============================================================================
# Result Aggregation Functions
# ============================================================================
def aggregate_predictions(
    text_result: Dict,
    audio_result: Dict,
    face_result: Dict
) -> Dict:
    """
    Aggregate predictions from all three models to determine final emotion.
    
    Args:
        text_result: Prediction from text model {"emotion": str, "confidence": float}
        audio_result: Prediction from audio model {"emotion": str, "confidence": float}
        face_result: Prediction from face model {"emotion": str, "confidence": float}
    
    Returns:
        Aggregated result with final emotion and analysis.
    """
    results = {
        'text': text_result,
        'audio': audio_result,
        'face': face_result
    }
    
    # Get all predicted emotions
    emotions = [
        text_result.get('emotion', 'unknown'),
        audio_result.get('emotion', 'unknown'),
        face_result.get('emotion', 'unknown')
    ]
    
    # Get confidences
    confidences = [
        text_result.get('confidence', 0.0),
        audio_result.get('confidence', 0.0),
        face_result.get('confidence', 0.0)
    ]
    
    # Check for unanimous agreement
    unique_emotions = set(e for e in emotions if e != 'unknown')
    
    if len(unique_emotions) == 1:
        # All models agree
        final_emotion = unique_emotions.pop()
        agreement = "unanimous"
        analysis = f"All three modalities detected '{final_emotion}' emotion."
    elif len(unique_emotions) == 2:
        # Majority vote or conflict
        from collections import Counter
        emotion_counts = Counter(e for e in emotions if e != 'unknown')
        most_common = emotion_counts.most_common(1)[0]
        
        if most_common[1] >= 2:
            # Majority (2/3) agree
            final_emotion = most_common[0]
            agreement = "majority"
            analysis = f"Majority (2/3) detected '{final_emotion}'. Possible mixed signals."
        else:
            # All different - use weighted average by confidence
            weighted_emotions = list(zip(emotions, confidences))
            weighted_emotions = [(e, c) for e, c in weighted_emotions if e != 'unknown']
            final_emotion = max(weighted_emotions, key=lambda x: x[1])[0]
            agreement = "conflict"
            analysis = "Conflicting signals detected. Using highest confidence prediction."
    else:
        # All different emotions
        weighted_emotions = list(zip(emotions, confidences))
        weighted_emotions = [(e, c) for e, c in weighted_emotions if e != 'unknown']
        if weighted_emotions:
            final_emotion = max(weighted_emotions, key=lambda x: x[1])[0]
        else:
            final_emotion = 'unknown'
        agreement = "conflict"
        analysis = "All modalities detected different emotions. Possible complex emotional state."
    
    # Detect potential sarcasm or deception
    special_cases = detect_special_cases(text_result, audio_result, face_result)
    if special_cases:
        analysis += f" Note: {special_cases}"
    
    return {
        'final_emotion': final_emotion,
        'agreement': agreement,
        'analysis': analysis,
        'individual_results': results,
        'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0
    }


def detect_special_cases(
    text_result: Dict,
    audio_result: Dict,
    face_result: Dict
) -> Optional[str]:
    """
    Detect special cases like sarcasm, deception, or mixed emotions.
    
    Returns:
        String describing the special case, or None.
    """
    text_emotion = text_result.get('emotion', 'unknown').lower()
    audio_emotion = audio_result.get('emotion', 'unknown').lower()
    face_emotion = face_result.get('emotion', 'unknown').lower()
    
    # Potential sarcasm: positive words but negative tone/expression
    positive_words = {'happy', 'joy', 'love', 'excited', 'grateful'}
    negative_expressions = {'angry', 'sad', 'disgust', 'fear', 'contempt'}
    
    if text_emotion in positive_words:
        if audio_emotion in negative_expressions or face_emotion in negative_expressions:
            return "Potential sarcasm detected (positive words with negative tone/expression)."
    
    # Masked emotion: neutral face but strong audio/text emotion
    if face_emotion == 'neutral':
        if audio_emotion in negative_expressions or text_emotion in negative_expressions:
            return "Possible suppressed/masked emotion detected."
    
    # Confusion: very low confidence across all modalities
    avg_conf = (
        text_result.get('confidence', 0) +
        audio_result.get('confidence', 0) +
        face_result.get('confidence', 0)
    ) / 3
    
    if avg_conf < 0.4:
        return "Low confidence across modalities - emotional state may be ambiguous."
    
    return None


# ============================================================================
# File Management Functions
# ============================================================================
def ensure_directories(base_path: str) -> None:
    """
    Ensure all required project directories exist.
    
    Args:
        base_path: Base path of the project.
    """
    directories = [
        'data/text_data',
        'data/audio_data',
        'data/face_data',
        'models/roberta_text',
        'models/wav2vec_audio',
        'models/resnet_face',
        'notebooks',
        'src',
    ]
    
    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Created/verified directory: {path}")


def get_model_path(model_type: str, base_path: str = '.') -> str:
    """
    Get the path to a saved model.
    
    Args:
        model_type: One of 'text', 'audio', 'face'.
        base_path: Base path of the project.
    
    Returns:
        Full path to the model directory.
    """
    model_dirs = {
        'text': 'models/roberta_text',
        'audio': 'models/wav2vec_audio',
        'face': 'models/resnet_face'
    }
    
    if model_type not in model_dirs:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(model_dirs.keys())}")
    
    return os.path.join(base_path, model_dirs[model_type])


if __name__ == "__main__":
    # Test the utility functions
    print("Utility functions loaded successfully!")
    print(f"Face emotion labels: {EMOTION_LABELS_FACE}")
    print(f"Audio emotion labels: {EMOTION_LABELS_AUDIO}")
    print(f"Target sample rate: {SAMPLE_RATE} Hz")
