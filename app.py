import re
import nltk
import uvicorn
import numpy as np
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware

from model.loader import load_cnn_model
from tokenizer.loader import load_tokenizer

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment prediction and word contribution analysis using CNN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← or ["https://your-frontend.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
model = load_cnn_model()
tokenizer = load_tokenizer()

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing functions
def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, '', text)

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    return ' '.join(ps.stem(word) for word in text.split())

def remove_stopwords(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return ' '.join(word for word in text.split() if word.lower() not in stopwords)

def preprocess_input_review(review):
    review = denoise_text(review)
    review = remove_special_characters(review)
    review = simple_stemmer(review)
    review = remove_stopwords(review)
    return review

def process_text(text):
    return text.lower()

def predict_proba(arr):
    processed = [process_text(i) for i in arr]
    list_tokenized_ex = tokenizer.texts_to_sequences(processed)
    Ex = pad_sequences(list_tokenized_ex, maxlen=500)
    pred = model.predict(Ex)
    return np.array([[1 - p[0], p[0]] for p in pred])

# Pydantic schema
class ReviewRequest(BaseModel):
    review: str

# Health check
@app.get("/")
def read_root():
    return {
        "STATUS": "OK",
        "MESSAGE": "Sentiment API is Running"
    }

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(data: ReviewRequest):
    review = data.review.strip()
    if not review:
        raise HTTPException(status_code=400, detail="Review text is required.")

    cleaned_review = preprocess_input_review(review)
    review_seq = tokenizer.texts_to_sequences([cleaned_review])
    review_padded = pad_sequences(review_seq, maxlen=500)

    prediction = model.predict(review_padded)[0][0]
    sentiment = "positive" if prediction >= 0.5 else "negative"

    return {
        "sentiment": sentiment,
        "score": float(prediction)
    }

# Explainability endpoint
@app.post("/analyze")
def analyze_contribution(data: ReviewRequest):
    review = data.review.strip()
    if not review:
        raise HTTPException(status_code=400, detail="Review text is required.")

    original_score = predict_proba([review])[0][1]

    words = review.split()
    contributions = []

    for i in range(len(words)):
        masked = words[:i] + ["<mask>"] + words[i + 1:]
        masked_text = " ".join(masked)
        masked_score = predict_proba([masked_text])[0][1]
        diff = original_score - masked_score
        contributions.append((words[i], diff))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_contributions = contributions[:10]

    return {
        "review": review,
        "explanation": [
            {"word": word, "contribution": round(float(score), 4)} for word, score in top_contributions
        ]
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
