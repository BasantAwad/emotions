"""
Multimodal Emotion AI - Main API / Integration Manager
Combines Text, Audio, and Face emotion analysis for comprehensive emotion detection.
"""

import os
import sys
import tempfile
from typing import Dict, Optional, List, Union
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.text_predictor import TextEmotionAnalyzer
from src.audio_predictor import AudioEmotionAnalyzer
from src.face_predictor import FaceEmotionAnalyzer
from src.utils import (
    extract_audio_from_video,
    extract_frames_from_video,
    split_video,
    aggregate_predictions
)

# Optional: Speech-to-Text using Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: OpenAI Whisper not installed. Text analysis from audio disabled.")
    print("Install with: pip install openai-whisper")


class MultimodalEmotionAnalyzer:
    """
    Main integration class that combines all three specialist models.
    Analyzes video/audio/text input and provides comprehensive emotion detection.
    """
    
    def __init__(
        self,
        text_model_path: Optional[str] = None,
        audio_model_path: Optional[str] = None,
        face_model_path: Optional[str] = None,
        use_pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize all three specialist models.
        
        Args:
            text_model_path: Path to fine-tuned text model.
            audio_model_path: Path to fine-tuned audio model.
            face_model_path: Path to fine-tuned face model.
            use_pretrained: If True, use pre-trained HuggingFace models.
            device: Device to run models on ('cuda' or 'cpu').
        """
        print("=" * 60)
        print("Initializing Multimodal Emotion Analysis System")
        print("=" * 60)
        
        # Initialize Text Analyzer (RoBERTa)
        print("\n[1/4] Loading Text Emotion Model (RoBERTa)...")
        self.text_analyzer = TextEmotionAnalyzer(
            model_path=text_model_path,
            use_pretrained=use_pretrained,
            device=device
        )
        
        # Initialize Audio Analyzer (Wav2Vec 2.0)
        print("\n[2/4] Loading Audio Emotion Model (Wav2Vec 2.0)...")
        self.audio_analyzer = AudioEmotionAnalyzer(
            model_path=audio_model_path,
            use_pretrained=use_pretrained,
            device=device
        )
        
        # Initialize Face Analyzer (ResNet50)
        print("\n[3/4] Loading Face Emotion Model (ResNet50)...")
        self.face_analyzer = FaceEmotionAnalyzer(
            model_path=face_model_path,
            device=device
        )
        
        # Initialize Speech-to-Text (Whisper)
        print("\n[4/4] Loading Speech-to-Text Model (Whisper)...")
        if WHISPER_AVAILABLE:
            try:
                self.stt_model = whisper.load_model("base")
                print("Whisper model loaded successfully!")
            except Exception as e:
                print(f"Error loading Whisper: {e}")
                self.stt_model = None
        else:
            self.stt_model = None
        
        print("\n" + "=" * 60)
        print("All models loaded! System ready for analysis.")
        print("=" * 60)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Transcribed text string.
        """
        if self.stt_model is None:
            print("Warning: Whisper not available. Returning empty transcription.")
            return ""
        
        try:
            result = self.stt_model.transcribe(audio_path)
            return result.get("text", "")
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for emotion.
        
        Args:
            text: Input text string.
        
        Returns:
            Emotion prediction dictionary.
        """
        return self.text_analyzer.predict(text)
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze audio for emotion (tone/prosody).
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Emotion prediction dictionary.
        """
        return self.audio_analyzer.predict(audio_path)
    
    def analyze_face(self, image_input: Union[str, 'np.ndarray']) -> Dict:
        """
        Analyze face image for emotion.
        
        Args:
            image_input: Path to image or numpy array.
        
        Returns:
            Emotion prediction dictionary.
        """
        return self.face_analyzer.predict(image_input)
    
    def analyze_video(
        self,
        video_path: str,
        frame_interval: float = 1.0,
        analyze_text_content: bool = True
    ) -> Dict:
        """
        Perform comprehensive multimodal analysis on a video file.
        
        This is the main entry point for video analysis. It:
        1. Extracts audio and frames from the video
        2. Analyzes facial expressions from frames
        3. Analyzes audio tone/prosody
        4. Transcribes speech and analyzes text content
        5. Aggregates all results for final emotion prediction
        
        Args:
            video_path: Path to the video file.
            frame_interval: Seconds between frame captures.
            analyze_text_content: If True, transcribe and analyze speech content.
        
        Returns:
            Comprehensive emotion analysis result.
        """
        print(f"\nAnalyzing video: {video_path}")
        print("-" * 50)
        
        results = {
            'video_path': video_path,
            'text_result': None,
            'audio_result': None,
            'face_result': None,
            'aggregated': None,
            'transcription': None
        }
        
        # Step 1: Extract audio and frames
        print("\n[Step 1] Extracting audio and frames...")
        try:
            audio_path, frames = split_video(video_path)
            print(f"  → Extracted {len(frames)} frames")
            print(f"  → Audio saved to: {audio_path}")
        except Exception as e:
            print(f"  → Error: {e}")
            return {'error': str(e), 'results': results}
        
        # Step 2: Analyze facial expressions
        print("\n[Step 2] Analyzing facial expressions...")
        if frames:
            # Use the middle frame or sample multiple frames
            sample_indices = [len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4]
            sample_indices = [i for i in sample_indices if i < len(frames)]
            
            if sample_indices:
                face_predictions = []
                for idx in sample_indices:
                    pred = self.face_analyzer.predict(frames[idx])
                    face_predictions.append(pred)
                
                # Get most confident prediction
                valid_preds = [p for p in face_predictions if p.get('confidence', 0) > 0]
                if valid_preds:
                    results['face_result'] = max(valid_preds, key=lambda x: x['confidence'])
                else:
                    results['face_result'] = face_predictions[0] if face_predictions else {
                        'emotion': 'unknown', 'confidence': 0.0
                    }
                
                print(f"  → Face emotion: {results['face_result']['emotion']} "
                      f"({results['face_result']['confidence']:.2%})")
        
        # Step 3: Analyze audio tone
        print("\n[Step 3] Analyzing audio tone/prosody...")
        results['audio_result'] = self.audio_analyzer.predict(audio_path)
        print(f"  → Audio emotion: {results['audio_result']['emotion']} "
              f"({results['audio_result']['confidence']:.2%})")
        
        # Step 4: Transcribe and analyze text content
        if analyze_text_content:
            print("\n[Step 4] Transcribing speech...")
            transcription = self.transcribe_audio(audio_path)
            results['transcription'] = transcription
            
            if transcription:
                print(f"  → Transcription: \"{transcription[:100]}...\"" if len(transcription) > 100 
                      else f"  → Transcription: \"{transcription}\"")
                
                print("\n[Step 5] Analyzing text content...")
                results['text_result'] = self.text_analyzer.predict(transcription)
                print(f"  → Text emotion: {results['text_result']['emotion']} "
                      f"({results['text_result']['confidence']:.2%})")
            else:
                print("  → No speech detected or transcription failed")
                results['text_result'] = {'emotion': 'neutral', 'confidence': 0.5}
        else:
            results['text_result'] = {'emotion': 'neutral', 'confidence': 0.5}
        
        # Step 5: Aggregate results
        print("\n[Step 6] Aggregating multimodal results...")
        results['aggregated'] = aggregate_predictions(
            text_result=results['text_result'] or {'emotion': 'unknown', 'confidence': 0.0},
            audio_result=results['audio_result'] or {'emotion': 'unknown', 'confidence': 0.0},
            face_result=results['face_result'] or {'emotion': 'unknown', 'confidence': 0.0}
        )
        
        # Clean up temp audio file
        try:
            os.remove(audio_path)
        except:
            pass
        
        # Print final result
        print("\n" + "=" * 50)
        print("FINAL RESULT")
        print("=" * 50)
        print(f"Detected Emotion: {results['aggregated']['final_emotion'].upper()}")
        print(f"Agreement Level: {results['aggregated']['agreement']}")
        print(f"Analysis: {results['aggregated']['analysis']}")
        print(f"Average Confidence: {results['aggregated']['average_confidence']:.2%}")
        
        return results
    
    def analyze_realtime_input(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        image_input: Optional[Union[str, 'np.ndarray']] = None
    ) -> Dict:
        """
        Analyze individual modality inputs (for real-time or API usage).
        
        Args:
            text: Text input (e.g., from speech transcription).
            audio_path: Path to audio file.
            image_input: Face image path or numpy array.
        
        Returns:
            Aggregated emotion analysis.
        """
        text_result = {'emotion': 'unknown', 'confidence': 0.0}
        audio_result = {'emotion': 'unknown', 'confidence': 0.0}
        face_result = {'emotion': 'unknown', 'confidence': 0.0}
        
        if text:
            text_result = self.analyze_text(text)
        
        if audio_path:
            audio_result = self.analyze_audio(audio_path)
            
            # Also transcribe for text analysis if no text provided
            if not text and self.stt_model:
                transcription = self.transcribe_audio(audio_path)
                if transcription:
                    text_result = self.analyze_text(transcription)
        
        if image_input is not None:
            face_result = self.analyze_face(image_input)
        
        aggregated = aggregate_predictions(text_result, audio_result, face_result)
        
        return {
            'text_result': text_result,
            'audio_result': audio_result,
            'face_result': face_result,
            'aggregated': aggregated
        }


# ============================================================================
# FastAPI REST API (Optional)
# ============================================================================
def create_api():
    """Create FastAPI application for REST API access."""
    try:
        from fastapi import FastAPI, UploadFile, File, Form
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        return None
    
    app = FastAPI(
        title="Multimodal Emotion AI API",
        description="Analyze emotions from text, audio, and facial expressions",
        version="1.0.0"
    )
    
    # Initialize analyzer (lazy loading)
    analyzer = None
    
    def get_analyzer():
        nonlocal analyzer
        if analyzer is None:
            analyzer = MultimodalEmotionAnalyzer()
        return analyzer
    
    @app.get("/")
    async def root():
        return {"message": "Multimodal Emotion AI API", "status": "running"}
    
    @app.post("/analyze/text")
    async def analyze_text(text: str = Form(...)):
        """Analyze emotion from text."""
        result = get_analyzer().analyze_text(text)
        return JSONResponse(content=result)
    
    @app.post("/analyze/audio")
    async def analyze_audio(file: UploadFile = File(...)):
        """Analyze emotion from audio file."""
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        result = get_analyzer().analyze_audio(temp_path)
        os.remove(temp_path)
        return JSONResponse(content=result)
    
    @app.post("/analyze/face")
    async def analyze_face(file: UploadFile = File(...)):
        """Analyze emotion from face image."""
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        result = get_analyzer().analyze_face(temp_path)
        os.remove(temp_path)
        return JSONResponse(content=result)
    
    @app.post("/analyze/video")
    async def analyze_video(file: UploadFile = File(...)):
        """Full multimodal analysis on video file."""
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        result = get_analyzer().analyze_video(temp_path)
        os.remove(temp_path)
        return JSONResponse(content=result)
    
    return app


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal Emotion AI System")
    parser.add_argument("--mode", choices=["demo", "api", "analyze"],
                        default="demo", help="Run mode")
    parser.add_argument("--video", type=str, help="Path to video file for analysis")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--image", type=str, help="Path to face image")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        # Run demo mode
        print("\n" + "=" * 60)
        print("MULTIMODAL EMOTION AI - DEMO MODE")
        print("=" * 60)
        
        # Initialize system
        analyzer = MultimodalEmotionAnalyzer()
        
        # Demo text analysis
        print("\n--- TEXT ANALYSIS DEMO ---")
        demo_texts = [
            "I'm so excited about this project!",
            "This is absolutely terrible news.",
            "I feel very calm and peaceful today."
        ]
        for text in demo_texts:
            result = analyzer.analyze_text(text)
            print(f"\"{text}\"")
            print(f"  → {result['emotion']} ({result['confidence']:.2%})\n")
        
        print("\nDemo complete! Use --mode analyze with --video/--text/--audio/--image for real analysis.")
    
    elif args.mode == "api":
        # Run API server
        print("Starting API server...")
        app = create_api()
        if app:
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    elif args.mode == "analyze":
        # Run analysis on provided input
        analyzer = MultimodalEmotionAnalyzer()
        
        if args.video:
            result = analyzer.analyze_video(args.video)
            print(f"\nFinal Emotion: {result['aggregated']['final_emotion']}")
        
        elif args.text or args.audio or args.image:
            result = analyzer.analyze_realtime_input(
                text=args.text,
                audio_path=args.audio,
                image_input=args.image
            )
            print(f"\nFinal Emotion: {result['aggregated']['final_emotion']}")
            print(f"Details: {result['aggregated']}")
        
        else:
            print("Please provide input: --video, --text, --audio, or --image")
