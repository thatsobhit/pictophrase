# 🧠 PictoPhrase: Smart Image Captioning with Voice

**PictoPhrase** is a smart assistive technology project designed to generate human-like captions for images using deep learning, and read them aloud using text-to-speech (TTS). It empowers users with visual or speech impairments by providing meaningful image interpretations and audio feedback.

---

## 🚀 Features
- 🖼️ **Image Feature Extraction** using **InceptionV3** CNN
- ✍️ **Caption Generation** via LSTM-based decoder
- 🔊 **Text-to-Speech (TTS)** using Google TTS (gTTS)
- 🧱 Modular architecture for easy customization and extension
- 🧪 Ready-to-train notebook and inference code included

---

## 📂 Project Structure
```bash
PictoPhrase/
├── app/
│   ├── __init__.py
│   ├── preprocess.py          # Image preprocessing
│   ├── feature_extractor.py   # CNN feature extraction
│   ├── model.py               # LSTM caption model
│   ├── inference.py           # Caption inference logic
│   ├── utils.py               # Token utilities
│   └── speech.py              # Text-to-speech (TTS)
├── data/
│   ├── captions.txt           # Captions file (for training)
│   ├── tokenizer.pkl          # Saved tokenizer
│   └── sample_images/
│       └── sample.jpg         # Image to caption
├── notebooks/
│   └── training.ipynb         # Model training notebook
├── model.h5                   # Trained captioning model (to be added)
├── main.py                    # Main inference script
├── requirements.txt           # Python dependencies
├── run.sh                     # Run the app (Linux/macOS)
└── README.md                  # This file
