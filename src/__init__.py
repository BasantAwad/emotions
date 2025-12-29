# Package initialization - Lazy imports to avoid dependency errors
# Import individual modules as needed instead of all at once

__all__ = ['TextEmotionAnalyzer', 'AudioEmotionAnalyzer', 'FaceEmotionAnalyzer']

def get_text_analyzer():
    from .text_predictor import TextEmotionAnalyzer
    return TextEmotionAnalyzer

def get_audio_analyzer():
    from .audio_predictor import AudioEmotionAnalyzer
    return AudioEmotionAnalyzer

def get_face_analyzer():
    from .face_predictor import FaceEmotionAnalyzer
    return FaceEmotionAnalyzer
