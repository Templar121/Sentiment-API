import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')

def load_tokenizer():
    return joblib.load(TOKENIZER_PATH)