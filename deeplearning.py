from tensorflow.keras.layers import Input, Dense, LayerNormalization, GaussianDropout, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredLogarithmicError as MSLE
import pandas_datareader.data as pdr
import datetime as dt


def get_stock_data():
    start_date = dt.datetime.now() - dt.timedelta(days=367)
    end_date = dt.datetime.now() - dt.timedelta(days=1)
    return pdr.DataReader(["AAPL", "FLNC"], "stooq", start_date, end_date)


def ClosingPriceModel(shape):
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

    closing_price_model = Model(inputs=input_layer, outputs=output)
    return closing_price_model


cpm = ClosingPriceModel((100, 3,))
cpm.compile(optimizer=Adam(learning_rate=0.001), loss=MSLE(), metrics=["accuracy"], run_eagerly=False, jit_compile=False, steps_per_execution=1)
