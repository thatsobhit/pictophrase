from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add, RepeatVector, Concatenate, Activation

def build_caption_model(vocab_size, max_length):
    # Feature input
    features_input = Input(shape=(2048,))
    features_dense = Dense(256, activation='relu')(features_input)
    features_repeat = RepeatVector(max_length)(features_dense)

    # Caption input
    captions_input = Input(shape=(max_length,))
    captions_embed = Embedding(vocab_size, 256, mask_zero=True)(captions_input)
    captions_lstm = LSTM(256, return_sequences=True)(captions_embed)

    # Combine features + captions with attention
    merged = Concatenate(axis=-1)([features_repeat, captions_lstm])
    attention = Dense(1, activation='tanh')(merged)
    attention = Activation('softmax')(attention)
    context = attention * captions_lstm
    context = LSTM(256)(context)

    output = Dense(256, activation='relu')(context)
    output = Dense(vocab_size, activation='softmax')(output)

    model = Model(inputs=[features_input, captions_input], outputs=output)
    return model
