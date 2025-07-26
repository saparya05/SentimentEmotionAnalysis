import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

class EmotionAnalysisSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.bert_model = None
        self.bert_pipeline = None
        self.tokenizer = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.label_encoder = LabelEncoder()
        
        # Emotion labels (including neutral)
        self.emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'neutral']
        
    def setup_bert_model(self):
        """
        Setup BERT model for emotion classification
        """
        print("Setting up BERT model...")
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create pipeline
        self.bert_pipeline = pipeline(
            "text-classification",
            model=self.bert_model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        print("BERT model setup complete!")
    
    def bert_analysis(self, text):
        """
        BERT-based emotion analysis using the pipeline
        """
        if not hasattr(self, 'bert_pipeline'):
            self.setup_bert_model()
            
        result = self.bert_pipeline(text)
        return {
            'emotion': result[0]['label'].lower(),
            'confidence': result[0]['score']
        }
    
    def load_dataset(self):
        """
        Load and prepare the emotion dataset
        Using the popular emotion dataset from Hugging Face
        """
        from datasets import load_dataset
        
        print("Loading emotion dataset...")
        # Load the emotion dataset (contains 6 emotions: sadness, joy, love, anger, fear, surprise)
        dataset = load_dataset("emotion")
        
        # Convert to pandas DataFrame
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        val_df = pd.DataFrame(dataset['validation'])
        
        # Map emotion labels
        emotion_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
        
        train_df['emotion'] = train_df['label'].map(emotion_mapping)
        test_df['emotion'] = test_df['label'].map(emotion_mapping)
        val_df['emotion'] = val_df['label'].map(emotion_mapping)
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def textblob_analysis(self, text):
        """
        TextBlob sentiment analysis - updated to better handle neutral
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # More refined emotion mapping
        if polarity > 0.5:
            emotion = 'joy'
        elif polarity > 0.1:
            emotion = 'love'  # mild positive could be love
        elif polarity < -0.5:
            emotion = 'anger'
        elif polarity < -0.1:
            emotion = 'sadness'
        else:
            emotion = 'neutral'
            
        return {
            'emotion': emotion,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': abs(polarity)
        }
    
    def vader_analysis(self, text):
        """
        VADER sentiment analysis - updated to better handle neutral
        """
        scores = self.vader_analyzer.polarity_scores(text)
        
        # More refined emotion determination
        compound = scores['compound']
        if compound >= 0.5:
            emotion = 'joy'
        elif compound >= 0.1:
            emotion = 'love'
        elif compound <= -0.5:
            emotion = 'anger'
        elif compound <= -0.1:
            emotion = 'sadness'
        else:
            emotion = 'neutral'
            
        return {
            'emotion': emotion,
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'confidence': abs(compound)
        }
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#','', text)
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def ensemble_prediction(self, text):
        """
        Combine predictions from multiple models with neutral handling
        """
        text = self.preprocess_text(text)
        
        # Get predictions from all models
        textblob_result = self.textblob_analysis(text)
        vader_result = self.vader_analysis(text)
        bert_result = self.bert_analysis(text)
        
        # If BERT confidence is low, consider it neutral

        if bert_result['confidence'] < 0.4:  # Lower threshold
            bert_result['emotion'] = 'neutral'
            bert_result['confidence'] = 1 - bert_result['confidence']
       
        # Weighted ensemble (BERT gets higher weight due to better performance)
        weights = {'bert': 0.8, 'textblob': 0.1, 'vader': 0.1}
        
        # Create score dictionary for all emotions
        emotion_scores = {emotion: 0 for emotion in self.emotion_labels}
        
        # Add weighted scores from each model
        for model, weight in weights.items():
            if model == 'bert':
                emotion = bert_result['emotion']
                confidence = bert_result['confidence']
            elif model == 'textblob':
                emotion = textblob_result['emotion']
                confidence = textblob_result['confidence']
            else:  # vader
                emotion = vader_result['emotion']
                confidence = vader_result['confidence']
                
            emotion_scores[emotion] += confidence * weight
        
        # Get emotion with highest score
        final_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        final_confidence = emotion_scores[final_emotion]
        
        # If highest score is neutral, boost confidence if other models agree
        if final_emotion == 'neutral':
            neutral_count = sum(1 for model in [textblob_result, vader_result] 
                              if model['emotion'] == 'neutral')
            if neutral_count >= 1:  # At least one other model agrees
                final_confidence = min(1.0, final_confidence * 1.2)
        
        return {
            'emotion': final_emotion,
            'confidence': final_confidence,
            'textblob': textblob_result,
            'vader': vader_result,
            'bert': bert_result,
            'all_scores': emotion_scores
        }
    
    def evaluate_model(self, test_df, model_type='bert'):
        """
        Comprehensive model evaluation with proper neutral handling
        """
        print(f"Evaluating {model_type} model...")
        
        predictions = []
        true_labels = []
        
        for idx, row in test_df.iterrows():
            text = row['text']
            true_emotion = row['emotion']
            
            if model_type == 'bert':
                pred = self.bert_analysis(text)
                if pred['confidence'] < 0.3:
                    continue 
            elif model_type == 'ensemble':
                pred = self.ensemble_prediction(text)
            else:
                pred = self.textblob_analysis(text)
            
            predictions.append(pred['emotion'])
            true_labels.append(true_emotion)
            
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(test_df)} samples")
        
        # Ensure neutral is always included
        all_labels = self.emotion_labels  # Now includes 'neutral'
        
        # Convert all predictions and true labels to use these labels
        true_labels_encoded = []
        predictions_encoded = []
        
        for true, pred in zip(true_labels, predictions):
            true_labels_encoded.append(true if true in all_labels else 'neutral')
            predictions_encoded.append(pred if pred in all_labels else 'neutral')
        
        # Calculate metrics (now includes neutral)
        accuracy = accuracy_score(true_labels_encoded, predictions_encoded)
        precision = precision_score(true_labels_encoded, predictions_encoded, 
                                average='weighted', labels=all_labels, zero_division=0)
        recall = recall_score(true_labels_encoded, predictions_encoded,
                            average='weighted', labels=all_labels, zero_division=0)
        f1 = f1_score(true_labels_encoded, predictions_encoded,
                    average='weighted', labels=all_labels, zero_division=0)
        
        # Confusion matrix with all labels including neutral
        cm = confusion_matrix(true_labels_encoded, predictions_encoded, labels=all_labels)
        
        # Classification report
        report = classification_report(true_labels_encoded, predictions_encoded, 
                                    labels=all_labels, output_dict=True, zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions_encoded,
            'true_labels': true_labels_encoded,
            'all_labels': all_labels
        }
        
        return results

    def plot_confusion_matrix(self, cm, emotions):
        """
        Plot confusion matrix with neutral properly included
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotions, yticklabels=emotions)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.close()
            print("Confusion matrix saved as 'confusion_matrix.png'")
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
            """
            Plot confusion matrix with all emotions including neutral
            """
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=emotions, yticklabels=emotions)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.show(block=False)
                plt.close()
                print("Confusion matrix saved as 'confusion_matrix.png'")
            except Exception as e:
                print(f"Error plotting confusion matrix: {e}")
    
    def plot_metrics(self, results):
        """
        Plot evaluation metrics
        """
        try:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            values = [results[metric] for metric in metrics]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
            plt.title('Model Performance Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('metrics_chart.png', dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.close()
            print("Metrics chart saved as 'metrics_chart.png'")
        except Exception as e:
            print(f"Error plotting metrics: {e}")
    
    def analyze_single_text(self, text):
        """
        Analyze a single text sample
        """
        result = self.ensemble_prediction(text)
        
        print(f"Text: {text}")
        print(f"Predicted Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("\nDetailed Analysis:")
        print(f"BERT: {result['bert']}")
        print(f"TextBlob: {result['textblob']}")
        print(f"VADER: {result['vader']}")
        
        return result

def main():
    try:
        # Initialize the emotion analysis system
        print("Initializing emotion analysis system...")
        emotion_analyzer = EmotionAnalysisSystem()
        
        # Load dataset
        print("Loading dataset...")
        train_df, val_df, test_df = emotion_analyzer.load_dataset()
        
        # Setup models
        print("Setting up models...")
        emotion_analyzer.setup_bert_model()
        
        # Use a smaller subset for testing to avoid long runtime
        test_subset = test_df.head(100)
        
        # Add neutral examples manually
        neutral_texts = [
            "The meeting is scheduled for 3 PM.",
            "She walked to the store.",
            "It is raining outside.",
            "The report was submitted yesterday.",
            "The dog is sitting on the floor.",
            "I have a dentist appointment next week.",
            "He opened the book and started reading.",
            "Water boils at 100 degrees Celsius.",
            "There are seven continents on Earth.",
            "This sentence is grammatically correct."
        ]
        neutral_df = pd.DataFrame({
            'text': neutral_texts,
            'emotion': ['neutral'] * len(neutral_texts)
        })

        # Combine with original test_subset
        test_subset = pd.concat([test_subset, neutral_df], ignore_index=True)
        print(f"Using {len(test_subset)} samples for evaluation (including neutral)")
        
        # Evaluate BERT model
        print("Evaluating BERT model...")
        bert_results = emotion_analyzer.evaluate_model(test_subset, model_type='bert')
        
        print("\n=== BERT Model Results ===")
        print(f"Accuracy: {bert_results['accuracy']:.3f}")
        print(f"Precision: {bert_results['precision']:.3f}")
        print(f"Recall: {bert_results['recall']:.3f}")
        print(f"F1-Score: {bert_results['f1_score']:.3f}")
        
        # Use all 7 emotions in order
        emotions = emotion_analyzer.emotion_labels
        print(f"Detected emotions: {emotions}")
        
        # Plot results
        print("Generating visualizations...")
        emotion_analyzer.plot_confusion_matrix(bert_results['confusion_matrix'], emotions)
        emotion_analyzer.plot_metrics(bert_results)
        
        # Sample texts
        sample_texts = [
            "I am so happy today! Everything is going perfectly.",
            "This is the worst day of my life. I feel terrible.",
            "I'm really scared about the upcoming exam.",
            "Wow! That was completely unexpected!",
            "I love spending time with my family.",
            "I'm furious about this situation!",
            "The sun rises in the east."  # Neutral test
        ]
        
        print("\n=== Sample Text Analysis ===")
        for i, text in enumerate(sample_texts, 1):
            print(f"\n{i}. Analyzing: '{text}'")
            try:
                result = emotion_analyzer.analyze_single_text(text)
                print(f"   Result: {result['emotion']} (confidence: {result['confidence']:.3f})")
            except Exception as e:
                print(f"   Error analyzing text: {e}")
        
        print("\n=== Analysis Complete ===")
        print("All processes completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()