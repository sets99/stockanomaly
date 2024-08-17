from tensorflow.keras.layers import Input, Dense, LayerNormalization, GaussianDropout, PReLU, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy as CCE
import pandas as pd


def TimeWasteModel(shape):
    input_layer = Input(shape=shape)
    norm_input = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(input_layer)

    hidden1 = Dense(32, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2")(norm_input)
    hidden1 = PReLU()(hidden1)
    hidden1 = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden1)
    hidden1 = GaussianDropout(0.6)(hidden1)

    hidden2 = Dense(64, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2")(hidden1)
    hidden2 = PReLU()(hidden2)
    hidden2 = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden2)
    hidden2 = GaussianDropout(0.4)(hidden2)

    hidden3 = Dense(32, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2")(hidden2)
    hidden3 = PReLU()(hidden3)
    hidden3 = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden3)

    output = Dense(1)(hidden3)
    output = Softmax()(output)

    time_waste_model = Model(inputs=input_layer, outputs=output)
    return time_waste_model


twl_df = pd.DataFrame(csv_dataObj)
twl_model = TimeWasteModel(shape=(49,))
twl_model.compile(optimizer=Adam(), loss=CCE(), run_eagerly=False)
twl_model.fit(x, y, epochs=16384, batch_size=32, validation_split=0.17, shuffle=True), 
