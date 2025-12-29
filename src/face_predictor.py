"""
Face Emotion Predictor - Specialist 3
Uses model fine-tuned on FER2013 dataset for facial emotion recognition.
Loads from HuggingFace: BasantAwad/facial-emotion
"""

import os
import cv2
import numpy as np
import torch
from typing import Dict, Optional, List, Union
from PIL import Image

# Try to import transformers for HuggingFace model
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")


class FaceEmotionAnalyzer:
    """
    Face-based emotion analyzer using HuggingFace model.
    Analyzes facial expressions to detect emotions.
    """
    
    # FER2013 emotion labels
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Image specifications
    TARGET_SIZE = (224, 224)
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_huggingface: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the Face Emotion Analyzer.
        
        Args:
            model_path: Path to locally saved model (optional).
            use_huggingface: If True, load model from HuggingFace.
            device: Device to run model on ('cuda' or 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if use_huggingface and HF_AVAILABLE:
            self._load_huggingface_model(model_path)
        elif model_path and os.path.exists(model_path):
            self._load_local_model(model_path)
        else:
            print("Warning: No model loaded. Please provide a model path or enable HuggingFace.")
    
    def _load_huggingface_model(self, model_path: Optional[str] = None):
        """Load model from HuggingFace Hub."""
        # Use custom model or default to user's uploaded model
        model_name = model_path or "BasantAwad/facial-emotion"
        print(f"Loading model from HuggingFace: {model_name}")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
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
            print(f"Error loading HuggingFace model: {e}")
            self.model = None
            self.processor = None
    
    def _load_local_model(self, model_path: str):
        """Load a locally saved model."""
        print(f"Loading local model from: {model_path}")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Local model loaded successfully!")
        except Exception as e:
            print(f"Error loading local model: {e}")
            self.model = None
    
    def _preprocess_image(self, image: np.ndarray) -> Image.Image:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR format from OpenCV).
        
        Returns:
            PIL Image ready for processing.
        """
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, self.TARGET_SIZE)
        
        # Convert to PIL Image
        return Image.fromarray(image)
    
    def _detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract face from image.
        
        Args:
            image: Input image.
        
        Returns:
            Cropped face image or None if no face detected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        padding = int(0.15 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w]
    
    def predict(
        self,
        image_input: Union[str, np.ndarray],
        detect_face: bool = True,
        return_all_scores: bool = False
    ) -> Dict:
        """
        Predict emotion from face image.
        
        Args:
            image_input: Path to image file or numpy array.
            detect_face: If True, detect and crop face first.
            return_all_scores: If True, return scores for all emotions.
        
        Returns:
            Dictionary with emotion and confidence.
        """
        if self.model is None or self.processor is None:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "face_detected": False,
                "error": "Model not loaded"
            }
        
        try:
            # Load image if path provided
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image from {image_input}")
            else:
                image = image_input.copy()
            
            # Detect face if requested
            face_detected = True
            if detect_face:
                face = self._detect_face(image)
                if face is not None:
                    image = face
                else:
                    face_detected = False
            
            # Preprocess
            pil_image = self._preprocess_image(image)
            
            # Process with HuggingFace processor
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predictions
            probs = probabilities[0].cpu().numpy()
            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])
            
            emotion = self.id2label.get(predicted_idx, self.EMOTION_LABELS[predicted_idx % len(self.EMOTION_LABELS)])
            
            response = {
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "face_detected": face_detected
            }
            
            if return_all_scores:
                response["all_scores"] = {
                    self.id2label.get(i, f"emotion_{i}"): round(float(p), 4)
                    for i, p in enumerate(probs)
                }
            
            return response
            
        except Exception as e:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "face_detected": False,
                "error": str(e)
            }
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        detect_face: bool = True
    ) -> List[Dict]:
        """Predict emotions for multiple images."""
        return [self.predict(img, detect_face=detect_face) for img in images]
    
    def predict_from_video_frames(
        self,
        frames: List[np.ndarray],
        detect_face: bool = True
    ) -> Dict:
        """Analyze emotions across video frames and aggregate results."""
        all_predictions = []
        emotion_counts = {}
        total_confidence = 0.0
        faces_detected = 0
        
        for frame in frames:
            result = self.predict(frame, detect_face=detect_face)
            all_predictions.append(result)
            
            if result['emotion'] != 'unknown':
                emotion = result['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_confidence += result['confidence']
                if result.get('face_detected', False):
                    faces_detected += 1
        
        valid_predictions = [p for p in all_predictions if p['emotion'] != 'unknown']
        
        if valid_predictions and emotion_counts:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            avg_confidence = total_confidence / len(valid_predictions)
        else:
            dominant_emotion = 'unknown'
            avg_confidence = 0.0
        
        return {
            "dominant_emotion": dominant_emotion,
            "average_confidence": round(avg_confidence, 4),
            "frames_analyzed": len(frames),
            "faces_detected": faces_detected,
            "emotion_distribution": emotion_counts,
            "frame_predictions": all_predictions
        }


# ============================================================================
# Testing / Demo
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Face Emotion Analyzer - Demo")
    print("=" * 60)
    
    # Initialize analyzer with HuggingFace model
    analyzer = FaceEmotionAnalyzer(use_huggingface=True)
    
    # Test with sample image (if available)
    test_image_path = "data/face_data/sample.jpg"
    
    if os.path.exists(test_image_path):
        print(f"\nAnalyzing: {test_image_path}")
        result = analyzer.predict(test_image_path, return_all_scores=True)
        print(f"  → Emotion: {result['emotion']}")
        print(f"  → Confidence: {result['confidence']:.2%}")
        print(f"  → Face detected: {result['face_detected']}")
        
        if 'all_scores' in result:
            print("  → All scores:")
            for emotion, score in sorted(result['all_scores'].items(), key=lambda x: -x[1]):
                print(f"      {emotion}: {score:.2%}")
    else:
        print(f"\nNo test image found at: {test_image_path}")
        print("To test, place an image file at the above path.")
    
    print("\nModel ready for inference!")
