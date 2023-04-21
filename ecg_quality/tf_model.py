import numpy as np

import model
import tensorflow as tf

class tf_model(model.Model):

    def __init__(self, model:str):

        self.model = tf.keras.models.load_model(model)

        # loads the model using the keras framework and it will be used later

    def process_ecg(self, signal: list):

        signal = np.expand_dims(signal, axis=0)

        self.model.predict(signal)



