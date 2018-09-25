from keras.layers import Dropout, BatchNormalization, Conv2D, Activation, Dense, Input, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.regularizers import l2
from sklearn.base import BaseEstimator
import numpy as np
 
class Regressor(BaseEstimator):
    def __init__(self):
        l2_weight = 0.0001
        model_in = Input(shape=(11, 11, 7)) 
        model = BatchNormalization()(model_in)
        model = Conv2D(64, (5, 5), padding="same")(model)
        model = Activation("relu")(model)
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        model = Conv2D(128, (3,3), padding="same")(model)
        model = Activation("relu")(model)
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        model = Conv2D(128, (3,3), padding="same")(model)
        model = Activation("relu")(model)
        model = Flatten()(model)
        model = Dense(256, kernel_regularizer=l2(l2_weight))(model)
        model = Activation("tanh")(model)
        model = Dense(128, kernel_regularizer=l2(l2_weight))(model)
        model = Dropout(0)(model)
        model = Activation("tanh")(model)
        model = Dense(1)(model)
        self.cnn_model = Model(model_in, model)
        self.cnn_model.compile(loss="mse", optimizer="adam")
        print(self.cnn_model.summary())
        return
 
    def fit(self, X, y): 
        self.cnn_model.fit(X, y, epochs=200, batch_size=64, verbose=1)
    
    def predict(self, X): 
        return self.cnn_model.predict(X).ravel()