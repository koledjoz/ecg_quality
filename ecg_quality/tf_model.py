import numpy as np
from utils import MODEL_PATH_DICT

import model
import tensorflow as tf


class tf_model(model.Model):

    def __init__(self, model:str):
        # steps_per_epoch = 10547
        # clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-6, maximal_learning_rate=1e-4,
        #                                           scale_fn=lambda x: 1.0, step_size=3 * steps_per_epoch)


        # Load the saved model from disk with the custom object scope
        self.model = tf.keras.models.load_model(MODEL_PATH_DICT[model], compile=False)


    def process_ecg(self, signal: list):

        signal = np.expand_dims(signal, axis=0)

        return self.model.predict(signal, verbose=0)

    def get_input_length(self):
        return self.model.get_config()['layers'][0]['config']['batch_input_shape'][1]

