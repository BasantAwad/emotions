"""
Text Emotion Predictor - Specialist 1
Uses RoBERTa model fine-tuned on GoEmotions dataset for text emotion classification.
Includes slang expansion for better informal text understanding.
"""

import os
import re
import torch
from typing import Dict, Optional, List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)


# ============================================================================
# Slang Expander - Converts informal text to contextual phrases
# ============================================================================
class SlangExpander:
    """
    Expands slang, abbreviations, and informal expressions to full contextual phrases.
    This helps the emotion classifier understand informal text better.
    """
    
    # Comprehensive slang dictionary organized by emotion category
    SLANG_DICT = {
        # ===== DISGUST =====
        'eww': 'that is disgusting',
        'ewww': 'that is really disgusting',
        'ewwww': 'that is absolutely disgusting',
        'ew': 'that is gross',
        'yuck': 'that is disgusting',
        'yucky': 'that is disgusting',
        'gross': 'that is disgusting and repulsive',
        'ick': 'that makes me feel disgusted',
        'bleh': 'that is unpleasant and disgusting',
        'barf': 'that makes me want to vomit from disgust',
        'puke': 'that is so disgusting it makes me sick',
        'nasty': 'that is disgusting and unpleasant',
        '🤢': 'that makes me feel sick and disgusted',
        '🤮': 'that is so disgusting I want to vomit',
        'ngl thats nasty': 'not going to lie that is disgusting',
        
        # ===== ANGER =====
        'ugh': 'I am frustrated and annoyed',
        'ughhh': 'I am really frustrated and annoyed',
        'argh': 'I am frustrated and angry',
        'grr': 'I am growling with anger',
        'grrr': 'I am very angry right now',
        'ffs': 'for fucks sake I am so angry',
        'smh': 'shaking my head in disappointment and frustration',
        'wtf': 'what the fuck I am shocked and angry',
        'wth': 'what the hell I am annoyed',
        'omfg': 'oh my fucking god I cannot believe this',
        'stfu': 'shut the fuck up I am angry',
        'gtfo': 'get the fuck out I am furious',
        'pissed': 'I am very angry and upset',
        'salty': 'I am bitter and resentful',
        'triggered': 'I am provoked and angry',
        'tf': 'what the fuck I am confused and annoyed',
        'bruh': 'I cannot believe this I am annoyed',
        'bro': 'I am exasperated with this situation',
        'fml': 'fuck my life I am so frustrated',
        '😤': 'I am frustrated and angry',
        '🤬': 'I am extremely angry and cursing',
        '😡': 'I am very angry',
        'im done': 'I am fed up and frustrated',
        "i can't": 'I am so frustrated I cannot handle this',
        'cant even': 'I am too frustrated to continue',
        
        # ===== HAPPINESS / JOY =====
        'lol': 'that is funny and I am laughing',
        'lmao': 'that is so funny I am laughing hard',
        'lmfao': 'that is hilarious I am dying of laughter',
        'rofl': 'that is so funny I am rolling on the floor laughing',
        'haha': 'that is funny and amusing',
        'hahaha': 'that is really funny',
        'hehe': 'that is amusing and I am giggling',
        'hihi': 'that is cute and amusing',
        'yay': 'I am excited and happy',
        'yayyy': 'I am very excited and joyful',
        'yaaas': 'yes I am thrilled and excited',
        'yasss': 'yes I am so happy and excited',
        'woohoo': 'I am celebrating and very happy',
        'woo': 'I am excited and happy',
        'omg': 'oh my god I am so excited',
        'yeet': 'I am excited and energetic',
        'lit': 'this is amazing and exciting',
        'fire': 'this is amazing and impressive',
        '🔥': 'this is amazing and impressive',
        'slay': 'that is amazing you did great',
        'slayed': 'you did amazingly well',
        'periodt': 'absolutely correct and I agree strongly',
        'bet': 'I agree and I am happy about this',
        'bussin': 'this is really good and enjoyable',
        'goat': 'greatest of all time this is amazing',
        'based': 'I strongly agree this is admirable',
        'w': 'this is a win I am happy',
        'dub': 'this is a win I am celebrating',
        'pog': 'that is amazing and exciting',
        'poggers': 'that is really exciting and cool',
        'lesgo': 'let us go I am excited',
        "let's go": 'I am excited and ready',
        'lfg': 'let us fucking go I am so excited',
        '😂': 'that is so funny I am crying laughing',
        '🤣': 'that is hilarious',
        '😊': 'I am happy and content',
        '😁': 'I am very happy and grinning',
        '🥳': 'I am celebrating and joyful',
        'ty': 'thank you I appreciate it',
        'tysm': 'thank you so much I really appreciate it',
        'thx': 'thanks I am grateful',
        'ily': 'I love you I care about you',
        'ilysm': 'I love you so much',
        
        # ===== SADNESS =====
        'oof': 'that is painful and I feel bad',
        'rip': 'rest in peace that is sad',
        'f': 'paying respects that is sad',
        'feels': 'I am experiencing strong sad emotions',
        'feelsbadman': 'that makes me feel sad',
        'tfw': 'that feeling when I am sad',
        'mfw': 'my face when I am disappointed',
        'sigh': 'I am feeling tired and sad',
        'meh': 'I am feeling indifferent and unhappy',
        'nvm': 'never mind I am disappointed',
        'welp': 'well that is unfortunate and sad',
        'brb crying': 'I am so sad I need to cry',
        'im crying': 'I am emotionally overwhelmed with sadness',
        '😢': 'I am sad and crying',
        '😭': 'I am crying very hard from sadness',
        '💔': 'my heart is broken I am sad',
        'L': 'this is a loss I am disappointed',
        
        # ===== FEAR / ANXIETY =====
        'ngl': 'not going to lie I am nervous',
        'lowkey': 'I am subtly feeling anxious',
        'highkey': 'I am openly feeling very anxious',
        'scared': 'I am feeling afraid',
        'shook': 'I am shocked and scared',
        'spooked': 'I am startled and afraid',
        'creeped out': 'I feel uncomfortable and scared',
        'sus': 'that seems suspicious and worrying',
        'sketchy': 'that seems suspicious and concerning',
        'yikes': 'that is concerning and makes me uncomfortable',
        'awks': 'this is awkward and uncomfortable',
        'awkward': 'this situation makes me uncomfortable',
        '😰': 'I am anxious and worried',
        '😨': 'I am scared and fearful',
        '😱': 'I am terrified and screaming',
        
        # ===== SURPRISE =====
        'whoa': 'I am surprised and amazed',
        'woah': 'I am surprised and impressed',
        'wow': 'I am amazed and surprised',
        'woww': 'I am really surprised and amazed',
        'no way': 'I cannot believe this I am shocked',
        'wait what': 'I am confused and surprised',
        'huh': 'I am confused and surprised',
        'wut': 'what I am confused and surprised',
        'wat': 'what I am surprised',
        'srsly': 'seriously I am surprised',
        'fr': 'for real I am surprised this is true',
        'frfr': 'for real for real I am genuinely surprised',
        'deadass': 'I am being completely serious and surprised',
        'cap': 'that is a lie I am in disbelief',
        'no cap': 'no lie I am being honest and amazed',
        'plot twist': 'I am surprised by this unexpected turn',
        '😮': 'I am surprised with my mouth open',
        '😲': 'I am astonished and shocked',
        '🤯': 'my mind is blown I am amazed',
        
        # ===== LOVE / AFFECTION =====
        'uwu': 'I am feeling cute and affectionate',
        'owo': 'I am curious and feeling cute',
        'aww': 'that is adorable and heartwarming',
        'awww': 'that is so cute and sweet',
        'cute': 'that is adorable I love it',
        'adorbs': 'that is adorable and I love it',
        'bae': 'my loved one my dear',
        'xoxo': 'hugs and kisses with love',
        '❤️': 'I love this with all my heart',
        '🥰': 'I am feeling loved and adoring',
        '😍': 'I am in love and admiring',
        '💕': 'I feel love and affection',
        
        # ===== BOREDOM / INDIFFERENCE =====
        'meh': 'I am feeling indifferent and bored',
        'idc': 'I do not care I am indifferent',
        'idgaf': 'I do not give a fuck I am indifferent',
        'whatever': 'I do not care about this',
        'k': 'okay I acknowledge but I am indifferent',
        'kk': 'okay I understand',
        'ig': 'I guess I am not very interested',
        'idek': 'I do not even know I am confused',
        'idk': 'I do not know',
        'tbh': 'to be honest',
        'imo': 'in my opinion',
        'imho': 'in my humble opinion',
    }
    
    # Patterns that indicate the text needs expansion
    SHORT_TEXT_THRESHOLD = 15  # Characters
    
    @classmethod
    def needs_expansion(cls, text: str) -> bool:
        """Check if text is short/slang and needs expansion."""
        clean_text = text.strip().lower()
        # Check if it's very short
        if len(clean_text) <= cls.SHORT_TEXT_THRESHOLD:
            return True
        # Check if it contains mainly emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        text_without_emoji = emoji_pattern.sub('', clean_text).strip()
        if len(text_without_emoji) <= 5:
            return True
        return False
    
    @classmethod
    def expand(cls, text: str) -> Tuple[str, bool]:
        """
        Expand slang in text to full contextual phrases.
        
        Returns:
            Tuple of (expanded_text, was_expanded)
        """
        original_text = text
        text_lower = text.strip().lower()
        
        # Direct lookup for exact matches
        if text_lower in cls.SLANG_DICT:
            return cls.SLANG_DICT[text_lower], True
        
        # Try to expand individual words/tokens
        expanded_parts = []
        was_expanded = False
        
        # Split by whitespace and punctuation but keep punctuation
        tokens = re.findall(r"[\w']+|[^\w\s]", text, re.UNICODE)
        
        for token in tokens:
            token_lower = token.lower()
            if token_lower in cls.SLANG_DICT:
                expanded_parts.append(cls.SLANG_DICT[token_lower])
                was_expanded = True
            else:
                expanded_parts.append(token)
        
        if was_expanded:
            expanded_text = ' '.join(expanded_parts)
            # Clean up spacing around punctuation
            expanded_text = re.sub(r'\s+([.,!?])', r'\1', expanded_text)
            return expanded_text, True
        
        # If no expansion happened but text is very short, add context
        if cls.needs_expansion(text) and not was_expanded:
            # Add neutral context for very short unknown text
            return f"The person said: {text}", True
        
        return original_text, False
    
    @classmethod
    def get_slang_emotion_hint(cls, text: str) -> Optional[str]:
        """
        Get a hint about what emotion the slang might indicate.
        """
        text_lower = text.strip().lower()
        
        # Check against known patterns
        disgust_patterns = ['ew', 'yuck', 'gross', 'ick', 'barf', 'nasty', '🤢', '🤮']
        anger_patterns = ['ugh', 'argh', 'grr', 'ffs', 'wtf', 'smh', '😤', '🤬', '😡']
        happy_patterns = ['lol', 'lmao', 'haha', 'yay', 'woo', '😂', '🤣', '😊']
        sad_patterns = ['oof', 'rip', 'sigh', '😢', '😭', '💔']
        fear_patterns = ['yikes', 'shook', 'sus', '😰', '😨', '😱']
        surprise_patterns = ['wow', 'whoa', 'omg', '😮', '😲', '🤯']
        
        for pattern in disgust_patterns:
            if pattern in text_lower:
                return 'disgust'
        for pattern in anger_patterns:
            if pattern in text_lower:
                return 'anger'
        for pattern in happy_patterns:
            if pattern in text_lower:
                return 'happy'
        for pattern in sad_patterns:
            if pattern in text_lower:
                return 'sad'
        for pattern in fear_patterns:
            if pattern in text_lower:
                return 'fear'
        for pattern in surprise_patterns:
            if pattern in text_lower:
                return 'surprise'
        
        return None


class TextEmotionAnalyzer:
    """
    Text-based emotion analyzer using RoBERTa.
    Analyzes the emotional content of text/transcribed speech.
    """
    
    # GoEmotions label mapping (28 labels)
    EMOTION_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # Mapping from fine-grained to basic emotions
    EMOTION_MAPPING = {
        'admiration': 'happy',
        'amusement': 'happy',
        'anger': 'angry',
        'annoyance': 'angry',
        'approval': 'happy',
        'caring': 'happy',
        'confusion': 'neutral',
        'curiosity': 'neutral',
        'desire': 'happy',
        'disappointment': 'sad',
        'disapproval': 'angry',
        'disgust': 'disgust',
        'embarrassment': 'fear',
        'excitement': 'happy',
        'fear': 'fear',
        'gratitude': 'happy',
        'grief': 'sad',
        'joy': 'happy',
        'love': 'happy',
        'nervousness': 'fear',
        'optimism': 'happy',
        'pride': 'happy',
        'realization': 'surprise',
        'relief': 'happy',
        'remorse': 'sad',
        'sadness': 'sad',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the Text Emotion Analyzer.
        
        Args:
            model_path: Path to locally saved model (if fine-tuned).
            use_pretrained: If True, use pre-trained model from HuggingFace.
            device: Device to run model on ('cuda' or 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if use_pretrained and model_path is None:
            # Use custom trained model from HuggingFace
            model_name = "BasantAwad/text-emotion-detction"
            print(f"Loading pre-trained model: {model_name}")
        else:
            model_name = model_path or "models/roberta_text"
            print(f"Loading local model from: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Also create a pipeline for easy inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1,
                top_k=None  # Return all labels with scores
            )
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to basic initialization...")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def predict(self, text: str, return_all_scores: bool = False, expand_slang: bool = True) -> Dict:
        """
        Predict emotion from text.
        
        Args:
            text: Input text string to analyze.
            return_all_scores: If True, return scores for all emotions.
            expand_slang: If True, expand slang/informal text before classification.
        
        Returns:
            Dictionary with emotion and confidence.
            {
                "emotion": "happy",
                "fine_grained_emotion": "joy",
                "confidence": 0.95,
                "original_text": "lol",
                "expanded_text": "that is funny and I am laughing",
                "slang_detected": True,
                "all_scores": {...}  # Optional
            }
        """
        if self.model is None:
            return {
                "emotion": "unknown",
                "fine_grained_emotion": "unknown",
                "confidence": 0.0,
                "error": "Model not loaded"
            }
        
        try:
            original_text = text
            slang_detected = False
            slang_hint = None
            
            # Expand slang if enabled
            if expand_slang:
                expanded_text, was_expanded = SlangExpander.expand(text)
                slang_detected = was_expanded
                slang_hint = SlangExpander.get_slang_emotion_hint(text)
                text_to_analyze = expanded_text
            else:
                text_to_analyze = text
            
            # Use pipeline for inference
            results = self.pipeline(text_to_analyze)[0]
            
            # Get top prediction
            top_result = max(results, key=lambda x: x['score'])
            fine_grained = top_result['label']
            confidence = top_result['score']
            
            # Map to basic emotion
            basic_emotion = self.EMOTION_MAPPING.get(fine_grained, 'neutral')
            
            # If we have a slang hint and model is uncertain, use the hint
            if slang_hint and confidence < 0.5:
                # Boost confidence if slang hint matches
                if slang_hint == basic_emotion or slang_hint == fine_grained:
                    confidence = min(confidence + 0.2, 0.85)
                else:
                    # Use slang hint as override for very low confidence
                    if confidence < 0.3:
                        basic_emotion = slang_hint
                        fine_grained = slang_hint
                        confidence = 0.7  # Moderate confidence from slang detection
            
            response = {
                "emotion": basic_emotion,
                "fine_grained_emotion": fine_grained,
                "confidence": round(confidence, 4),
                "original_text": original_text,
                "slang_detected": slang_detected
            }
            
            if slang_detected:
                response["expanded_text"] = text_to_analyze
            
            if slang_hint:
                response["slang_emotion_hint"] = slang_hint
            
            if return_all_scores:
                response["all_scores"] = {
                    r['label']: round(r['score'], 4) for r in results
                }
            
            return response
            
        except Exception as e:
            return {
                "emotion": "unknown",
                "fine_grained_emotion": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict emotions for multiple texts.
        
        Args:
            texts: List of text strings.
        
        Returns:
            List of prediction dictionaries.
        """
        return [self.predict(text) for text in texts]
    
    def get_emotion_distribution(self, text: str) -> Dict[str, float]:
        """
        Get the full distribution of emotion probabilities.
        
        Args:
            text: Input text string.
        
        Returns:
            Dictionary mapping each emotion to its probability.
        """
        result = self.predict(text, return_all_scores=True)
        return result.get("all_scores", {})
    
    def analyze_sentiment_aspects(self, text: str) -> Dict:
        """
        Perform detailed sentiment analysis on the text.
        
        Args:
            text: Input text string.
        
        Returns:
            Detailed analysis including dominant emotions and sentiment.
        """
        all_scores = self.get_emotion_distribution(text)
        
        if not all_scores:
            return {"error": "Could not analyze text"}
        
        # Group by basic emotion categories
        basic_emotions = {}
        for emotion, score in all_scores.items():
            basic = self.EMOTION_MAPPING.get(emotion, 'neutral')
            if basic not in basic_emotions:
                basic_emotions[basic] = 0.0
            basic_emotions[basic] += score
        
        # Normalize
        total = sum(basic_emotions.values())
        if total > 0:
            basic_emotions = {k: v/total for k, v in basic_emotions.items()}
        
        # Determine overall sentiment
        positive_emotions = {'happy', 'surprise'}
        negative_emotions = {'sad', 'angry', 'fear', 'disgust'}
        
        positive_score = sum(basic_emotions.get(e, 0) for e in positive_emotions)
        negative_score = sum(basic_emotions.get(e, 0) for e in negative_emotions)
        neutral_score = basic_emotions.get('neutral', 0)
        
        if positive_score > negative_score and positive_score > neutral_score:
            overall_sentiment = 'positive'
        elif negative_score > positive_score and negative_score > neutral_score:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            "fine_grained_emotions": all_scores,
            "basic_emotions": basic_emotions,
            "overall_sentiment": overall_sentiment,
            "sentiment_scores": {
                "positive": round(positive_score, 4),
                "negative": round(negative_score, 4),
                "neutral": round(neutral_score, 4)
            }
        }


# ============================================================================
# Testing / Demo
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Text Emotion Analyzer - Demo")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TextEmotionAnalyzer(use_pretrained=True)
    
    # Test sentences
    test_texts = [
        "I'm so happy today! Everything is going great!",
        "I can't believe they would do such a thing. I'm furious!",
        "I feel really sad about what happened.",
        "Oh wow, I didn't expect that at all!",
        "The weather is okay today, nothing special.",
        "I'm grateful for all the help you've given me.",
        "That's absolutely disgusting behavior.",
        "I'm really worried about the upcoming exam."
    ]
    
    print("\nAnalyzing sample texts...\n")
    
    for text in test_texts:
        result = analyzer.predict(text)
        print(f"Text: \"{text}\"")
        print(f"  → Emotion: {result['emotion']} ({result['fine_grained_emotion']})")
        print(f"  → Confidence: {result['confidence']:.2%}")
        print()
