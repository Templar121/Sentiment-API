import inspect
if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        full = inspect.getfullargspec(func)
        return (full.args, full.varargs, full.varkw, full.defaults)
    inspect.getargspec = getargspec




import re
import nltk
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import numpy as np


from model.loader import load_cnn_model
from tokenizer.loader import load_tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load your trained model and tokenizer
model = load_cnn_model()
tokenizer = load_tokenizer()

# Download NLTK resources if not already
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
    text = re.sub(pattern, '', text)
    return text


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

class_names = ['negative', 'positive']
# explainer = shap.Explainer(model.predict, masker=shap.maskers.Text(tokenizer))

def process_text(text):
    return text.lower()

def predict_proba(arr):
    processed = [process_text(i) for i in arr]
    list_tokenized_ex = tokenizer.texts_to_sequences(processed)
    Ex = pad_sequences(list_tokenized_ex, maxlen=500)  # match max_review_length
    pred = model.predict(Ex)
    return np.array([[1 - p[0], p[0]] for p in pred])

# Health check endpoint
@app.route('/')
def index():
    return jsonify({
        'STATUS': 'OK',
        'MESSAGE': 'Sentiment API is Running',
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review = data.get('review', '')

    # Preprocess the review
    cleaned_review = preprocess_input_review(review)

    # Tokenize and pad sequence
    review_seq = tokenizer.texts_to_sequences([cleaned_review])
    maxlen = 500  # ensure this matches training
    review_padded = pad_sequences(review_seq, maxlen=maxlen)

    # Predict sentiment
    prediction = model.predict(review_padded)[0][0]
    sentiment = 'positive' if prediction >= 0.5 else 'negative'

    # Return JSON response
    return jsonify({
        'sentiment': sentiment,
        'score': float(prediction)
    })
    
    
# @app.route('/analyze', methods=['POST'])
# def analyze():
#     data = request.get_json(force=True)
#     review = data.get('review', '')

#     if not review.strip():
#         return jsonify({'error': 'Review text is required.'}), 400

#     # Generate LIME explanation
#     explanation = explainer.explain_instance(review, predict_proba, num_features=10)

#     # Extract important features
#     important_words = explanation.as_list()

#     # Structure the response
#     response = {
#         'review': review,
#         'explanation': [{'word': word, 'contribution': float(score)} for word, score in important_words]
#     }

#     return jsonify(response)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    review = data.get('review', '')

    if not review.strip():
        return jsonify({'error': 'Review text is required.'}), 400

    original_score = predict_proba([review])[0][1]  # e.g., positive class

    words = review.split()
    contributions = []

    for i in range(len(words)):
        masked = words[:i] + ["<mask>"] + words[i+1:]
        masked_text = " ".join(masked)
        masked_score = predict_proba([masked_text])[0][1]
        diff = original_score - masked_score
        contributions.append((words[i], diff))

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_contributions = contributions[:10]

    response = {
        'review': review,
        'explanation': [{'word': word, 'contribution': round(float(score), 4)} for word, score in top_contributions]
    }

    return jsonify(response)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
