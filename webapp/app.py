from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from app.preprocess import load_and_preprocess_image
from app.feature_extractor import build_feature_extractor
from app.inference import generate_caption
from app.speech import text_to_speech
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and tokenizer once
model = load_model('model.h5')
fe_model = build_feature_extractor()
with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 34

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        if img:
            filename = secure_filename(img.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(path)

            # Process image
            processed = load_and_preprocess_image(path)
            feature = fe_model.predict(processed)
            caption = generate_caption(model, tokenizer, feature, MAX_LENGTH)
            text_to_speech(caption)

            return render_template('index.html', image_path=path, caption=caption)

    return render_template('index.html', image_path=None, caption=None)

if __name__ == '__main__':
    app.run(debug=True)
