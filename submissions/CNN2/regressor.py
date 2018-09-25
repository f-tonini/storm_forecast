from keras.layers import Conv2D, Activation, Dense, Input, MaxPooling2D, Flatten, Dropout, SeparableConv2D, Concatenate
from keras.models import Model
from keras.regularizers import l2
from sklearn.base import BaseEstimator
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        l2_weight = 0.0001
        model_in = Input(shape=(11, 11, 5))
        scalar_in = Input(shape=(5,))
        model = SeparableConv2D(32, (3, 3), padding="same")(model_in)
        model = Activation("relu")(model)
        model = MaxPooling2D()(model)
        model = Conv2D(40, (3,3), padding="same")(model)
        model = Activation("relu")(model)
        model = MaxPooling2D()(model)
        model = Conv2D(48, (3, 3), padding="same")(model)
        model = Activation("relu")(model)
        model = Flatten()(model)
        model = Concatenate()([model, scalar_in])
        model = Dense(256)(model)
        model = Activation("selu")(model)
        model = Dense(256)(model)
        model = Activation("selu")(model)
        model = Dense(1)(model)
        self.cnn_model = Model([model_in, scalar_in], model)
        self.cnn_model.compile(loss="mse", optimizer="adam")
        print(self.cnn_model.summary())
        return

    def fit(self, X, y):
        self.cnn_model.fit(X, y, epochs=30, batch_size=128, verbose=1)

    def predict(self, X):
        return self.cnn_model.predict(X).ravel()