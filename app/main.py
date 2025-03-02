from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import pickle
import joblib
from fastapi.responses import HTMLResponse

# Load the trained model and tokenizer
model = joblib.load('app/model.pkl')

with open('app/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define constants
max_review_length = 500
class_names = ['negative', 'positive']
explainer = LimeTextExplainer(class_names=class_names)

# FastAPI app instance
app = FastAPI(title="Sentiment Analysis with LIME Explainer")

# Define request model
class InputText(BaseModel):
    text: str

def process_text(text):
    return text.lower()

def predict_proba(arr):
    processed = [process_text(i) for i in arr]
    list_tokenized_ex = tokenizer.texts_to_sequences(processed)
    Ex = pad_sequences(list_tokenized_ex, maxlen=max_review_length)
    pred = model.predict(Ex)
    return np.array([[1 - i[0], i[0]] for i in pred])

def explain_text(text):
    explanation = explainer.explain_instance(text, predict_proba, num_features=10)
    return explanation

@app.post("/analyze", response_class=HTMLResponse)
def analyze_sentiment(input_text: InputText):
    """
    Endpoint to analyze sentiment and provide LIME explanation.
    """
    if not input_text.text:
        raise HTTPException(status_code=400, detail="Input text is required.")
    
    # Predict sentiment
    prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences([input_text.text]), maxlen=max_review_length))
    sentiment = class_names[int(prediction[0][0] > 0.5)]
    
    # Generate LIME explanation
    explanation = explain_text(input_text.text)
    explanation_html = explanation.as_html()

    # Combine sentiment and explanation in HTML format
    response_html = f"""
    <html>
        <body>
            <h2>Predicted Sentiment: {sentiment.capitalize()}</h2>
            <h3>LIME Explanation:</h3>
            {explanation_html}
        </body>
    </html>
    """
    return HTMLResponse(content=response_html)

@app.get("/", response_class=HTMLResponse)
def home():
    """
    Home endpoint with instructions for using the API.
    """
    return """
    <html>
        <head>
            <title>Sentiment Analysis API</title>
        </head>
        <body>
            <h1>Sentiment Analysis with LIME Explanation</h1>
            <p>Use the <code>/analyze</code> endpoint to perform sentiment analysis.</p>
            <p>Send a POST request with JSON payload:</p>
            <pre>
{
    "text": "Your input text here"
}
            </pre>
            <p>The response will include the predicted sentiment and an HTML explanation.</p>
        </body>
    </html>
    """




