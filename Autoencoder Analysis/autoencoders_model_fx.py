import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.optimizers import Adam
from keras import backend as K




def encoder_fx1(n_hidden_nodes_1=20,n_hidden_nodes_2=10, n_cols=142, dropout_p = 0.5):

    input = Input(shape=(n_cols, ))
    encode_1 = Dense(n_hidden_nodes_1, input_shape = (n_cols,), activation="relu")(input)
    drop = Dropout(dropout_p, seed=1234)(encode_1)
    bn = BatchNormalization()(drop)

    encoded = Dense(n_hidden_nodes_2, input_shape = (n_cols,), activation="relu")(bn)
    drop = Dropout(dropout_p, seed=123)(encoded)
    bn = BatchNormalization()(drop)

    decoded = Dense(n_hidden_nodes, input_shape = (n_cols,), activation="relu")(bn)

    model = Model(input, decoded)

    return(model)

def encoder_fx2(n_hidden_nodes_1 = 10, dropout_p = 0.3, n_cols = 142):

    input = Input(shape=(n_cols, ))
    x = Dense(n_hidden_nodes_1, activation="relu")(input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_p, seed=1234)(x)

    output = Dense(n_cols, activation="linear")(x)
    model = Model(input, output)

    return(model)

def col_pct_diff(y_true, y_pred):
    diff = (y_true, y_pred)
    pct = diff/y_true
    avg_diff = K.mean(pct)
    return(avg_diff)
