import pickle
import numpy as np
from tensorflow.keras.models import load_model
from app.preprocess import load_and_preprocess_image
from app.feature_extractor import build_feature_extractor
from app.inference import generate_caption
from app.speech import text_to_speech

IMG_PATH = 'data/sample_images/sample.jpg'
MODEL_PATH = 'model.h5'
TOKENIZER_PATH = 'data/tokenizer.pkl'
MAX_LENGTH = 34  # Update as per your training

def main():
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = load_model(MODEL_PATH)
    image = load_and_preprocess_image(IMG_PATH)

    fe_model = build_feature_extractor()
    photo = fe_model.predict(image)

    caption = generate_caption(model, tokenizer, photo, MAX_LENGTH)
    print("Caption:", caption)

    text_to_speech(caption)

if name == 'main':
    main()
