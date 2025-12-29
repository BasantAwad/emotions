"""
Audio Emotion Predictor - Specialist 2
Uses Wav2Vec 2.0 model fine-tuned on RAVDESS dataset for speech emotion recognition.
"""

import os
import torch
import numpy as np
import librosa
from typing import Dict, Optional, List, Tuple
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)


class AudioEmotionAnalyzer:
    """
    Audio-based emotion analyzer using Wav2Vec 2.0.
    Analyzes the emotional content of speech audio (tone, pitch, rhythm).
    """
    
    # RAVDESS emotion labels
    EMOTION_LABELS = [
        'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    # Mapping to standardized emotion labels
    EMOTION_MAPPING = {
        'neutral': 'neutral',
        'calm': 'neutral',
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'fearful': 'fear',
        'disgust': 'disgust',
        'surprised': 'surprise'
    }
    
    # Target sample rate for Wav2Vec 2.0
    SAMPLE_RATE = 16000
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the Audio Emotion Analyzer.
        
        Args:
            model_path: Path to locally saved model (if fine-tuned).
            use_pretrained: If True, use pre-trained model from HuggingFace.
            device: Device to run model on ('cuda' or 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if use_pretrained and model_path is None:
            # Use pre-trained speech emotion recognition model
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            # Alternative: "superb/wav2vec2-base-superb-er"
            print(f"Loading pre-trained model: {model_name}")
        else:
            model_name = model_path or "models/wav2vec_audio"
            print(f"Loading local model from: {model_name}")
        
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get label mapping from model config
            if hasattr(self.model.config, 'id2label'):
                self.id2label = self.model.config.id2label
            else:
                self.id2label = {i: label for i, label in enumerate(self.EMOTION_LABELS)}
            
            print(f"Model loaded successfully on {self.device}")
            print(f"Labels: {list(self.id2label.values())}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to basic initialization...")
            self.model = None
            self.feature_extractor = None
            self.id2label = {}
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to the audio file.
        
        Returns:
            Tuple of (waveform, sample_rate)
        """
        # Load audio with librosa
        waveform, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE)
        
        # Normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        return waveform, sr
    
    def _preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio waveform for the model.
        
        Args:
            waveform: Audio waveform as numpy array.
        
        Returns:
            Preprocessed tensor ready for model input.
        """
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        return inputs.input_values.to(self.device)
    
    def predict(self, audio_path: str, return_all_scores: bool = False) -> Dict:
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.)
            return_all_scores: If True, return scores for all emotions.
        
        Returns:
            Dictionary with emotion and confidence.
            {
                "emotion": "angry",
                "raw_emotion": "angry",
                "confidence": 0.88,
                "all_scores": {...}  # Optional
            }
        """
        if self.model is None:
            return {
                "emotion": "unknown",
                "raw_emotion": "unknown",
                "confidence": 0.0,
                "error": "Model not loaded"
            }
        
        try:
            # Load and preprocess audio
            waveform, _ = self._load_audio(audio_path)
            input_values = self._preprocess(waveform)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_values)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predictions
            probs = probabilities[0].cpu().numpy()
            predicted_id = np.argmax(probs)
            confidence = float(probs[predicted_id])
            
            raw_emotion = self.id2label.get(predicted_id, 'unknown')
            basic_emotion = self.EMOTION_MAPPING.get(raw_emotion.lower(), 'neutral')
            
            response = {
                "emotion": basic_emotion,
                "raw_emotion": raw_emotion,
                "confidence": round(confidence, 4)
            }
            
            if return_all_scores:
                response["all_scores"] = {
                    self.id2label.get(i, f'label_{i}'): round(float(p), 4)
                    for i, p in enumerate(probs)
                }
            
            return response
            
        except Exception as e:
            return {
                "emotion": "unknown",
                "raw_emotion": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def predict_from_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict:
        """
        Predict emotion from audio waveform directly.
        
        Args:
            waveform: Audio waveform as numpy array.
            sample_rate: Sample rate of the waveform.
        
        Returns:
            Dictionary with emotion and confidence.
        """
        if self.model is None:
            return {
                "emotion": "unknown",
                "raw_emotion": "unknown",
                "confidence": 0.0,
                "error": "Model not loaded"
            }
        
        try:
            # Resample if necessary
            if sample_rate != self.SAMPLE_RATE:
                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=self.SAMPLE_RATE
                )
            
            # Normalize
            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform))
            
            # Preprocess and predict
            input_values = self._preprocess(waveform)
            
            with torch.no_grad():
                outputs = self.model(input_values)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            probs = probabilities[0].cpu().numpy()
            predicted_id = np.argmax(probs)
            confidence = float(probs[predicted_id])
            
            raw_emotion = self.id2label.get(predicted_id, 'unknown')
            basic_emotion = self.EMOTION_MAPPING.get(raw_emotion.lower(), 'neutral')
            
            return {
                "emotion": basic_emotion,
                "raw_emotion": raw_emotion,
                "confidence": round(confidence, 4)
            }
            
        except Exception as e:
            return {
                "emotion": "unknown",
                "raw_emotion": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_audio_features(self, audio_path: str) -> Dict:
        """
        Extract and analyze audio features (pitch, energy, etc.)
        
        Args:
            audio_path: Path to the audio file.
        
        Returns:
            Dictionary with audio feature analysis.
        """
        try:
            waveform, sr = self._load_audio(audio_path)
            
            # Extract features
            # Pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
            pitch_values = pitches[pitches > 0]
            
            # Energy/RMS
            rms = librosa.feature.rms(y=waveform)[0]
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
            
            # Zero crossing rate (indicator of noisiness)
            zcr = librosa.feature.zero_crossing_rate(waveform)[0]
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
            
            return {
                "duration_seconds": len(waveform) / sr,
                "pitch": {
                    "mean": float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0,
                    "std": float(np.std(pitch_values)) if len(pitch_values) > 0 else 0,
                    "max": float(np.max(pitch_values)) if len(pitch_values) > 0 else 0
                },
                "energy": {
                    "mean": float(np.mean(rms)),
                    "std": float(np.std(rms)),
                    "max": float(np.max(rms))
                },
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "zero_crossing_rate_mean": float(np.mean(zcr)),
                "mfcc_means": [float(np.mean(mfcc)) for mfcc in mfccs]
            }
            
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# Testing / Demo
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Audio Emotion Analyzer - Demo")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = AudioEmotionAnalyzer(use_pretrained=True)
    
    # Test with sample audio (if available)
    test_audio_path = "data/audio_data/sample.wav"
    
    if os.path.exists(test_audio_path):
        print(f"\nAnalyzing: {test_audio_path}")
        result = analyzer.predict(test_audio_path, return_all_scores=True)
        print(f"  → Emotion: {result['emotion']} ({result['raw_emotion']})")
        print(f"  → Confidence: {result['confidence']:.2%}")
        
        if 'all_scores' in result:
            print("  → All scores:")
            for emotion, score in sorted(result['all_scores'].items(), key=lambda x: -x[1]):
                print(f"      {emotion}: {score:.2%}")
    else:
        print(f"\nNo test audio file found at: {test_audio_path}")
        print("To test, place a WAV file at the above path.")
    
    print("\nModel ready for inference!")
