import os
from tensorflow.keras.models import load_model  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')

def load_cnn_model():
    return load_model(MODEL_PATH)  # ✅ Correct usage