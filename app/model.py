from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model

def build_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model
#feature-extractor
