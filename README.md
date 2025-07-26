# 🧠 Sentiment and Emotion Analysis System

This project implements a hybrid **Sentiment and Emotion Analysis System** that combines traditional lexicon-based methods (TextBlob, VADER) with a transformer-based deep learning model (DistilRoBERTa from Hugging Face).

The system predicts both **emotion categories** (`anger`, `fear`, `joy`, `love`, `sadness`, `surprise`, `neutral`) and **sentiment polarity** (positive, negative, neutral) using a weighted ensemble strategy for improved accuracy.

---

## 🚀 Features

- Lexicon-based sentiment analysis using **TextBlob** and **VADER**
- Emotion classification using **DistilRoBERTa** (`j-hartmann/emotion-english-distilroberta-base`)
- Weighted ensemble method for combining model predictions
- Robust detection of **neutral** sentiment/emotion
- Evaluation using accuracy, precision, recall, F1-score, confusion matrix
- Clean, modular code with preprocessing, prediction, and evaluation

---

## 🧰 Technologies Used

- Python
- TextBlob
- VADER Sentiment
- Hugging Face Transformers
- scikit-learn
- matplotlib & seaborn
- Numpy
- Pandas
- Tensonflow
- Pytorch

---

## 🛠️ Setup & Usage

1. **Clone the Repository**

```bash
git clone https://github.com/saparya05/Employee-Management-System
cd sentiment-emotion-analysis
```
2. **Activate Virtual Environment**
```bash
# Linux/macOS
source env/bin/activate

# Windows
.\env\Scripts\activate
```
3. **Install Dependencies**

```bash
pip install -r requirements.txt
```
4. **Run the Script**
```bash
python main.py
```

## 📊 Sample Output

sample_texts = [I am so happy today! Everything is going perfectly."]
```bash
Analyzing: 'I am so happy today! Everything is going perfectly.'
Text: I am so happy today! Everything is going perfectly.
Predicted Emotion: joy
Confidence: 0.972
```

## 📁 Dataset
This project uses the Emotion Dataset from Hugging Face's datasets library.
Contains 6 emotion labels: sadness, joy, love, anger, fear, and surprise
Automatically loaded using:

```bash
from datasets import load_dataset
dataset = load_dataset("emotion")
```
## 📌 License
This project is open-source and available under the MIT License.
