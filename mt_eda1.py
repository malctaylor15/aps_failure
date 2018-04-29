import pandas as pd
import numpy as np
import requests

#Data from https://archive.ics.uci.edu/ml/machine-learning-databases/00421/

#data_raw = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv", skiprows = 20)
data_raw = pd.read_csv("aps_failure_training_set.csv", skiprows = 20)

# Try to understand data
data_raw.shape
data_raw.head()


# How many columns have the same number of characters
pd.Series([len(col) for col in data_raw.columns]).value_counts()
# Most columns have 6 characters... lets see if we can group by somehow

# Which cols don't have 6 characters
[(col, len(col)) for col in data_raw.columns if len(col) != 6]
# 3, one id the dependent (class), others follow the same structure

# Compare first 2 characters
len(set([col[:2] for col in data_raw.columns]))

# Quick look for categorical variables
counts = {col:data_raw[col].nunique() for col in data_raw.columns if data_raw[col].nunique() <100}
counts


"""

This is in clean_data1.py

pct_na_dict = {col:data_raw[col].value_counts(normalize=True).loc["na"] for col in data_raw.columns if "na" in data_raw[col].value_counts().index}
len(pct_na)

pct_na = pd.Series(pct_na_dict)
pct_na.describe()
large_pct_na = pct_na[pct_na > 0.15].index
len(large_pct_na)

data_raw.drop(large_pct_na, axis = 1, inplace= True)

data_temp2 = data_raw.iloc[:1000, :50]

max(data1.values.tolist())

# Replace "na"s with the mean -- look into
data1 = data_raw.replace(["na", "nan"], [data_raw.mean(), data_raw.mean()])
# data2 = data1.fillna(data_raw.mean())

#([x for x in data2.values.tolist() if "nan" == x])
max(data1.values.tolist())

# Start splits
X = data1.drop(["class"], axis = 1)
Y = data_raw["class"]


"""
from clean_data1 import clean_data
X, Y = clean_data(data_raw)

Y.value_counts(normalize=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state = 1)

compare_samples = pd.DataFrame({"Train":y_train.value_counts(normalize=True),
"Test": y_test.value_counts(normalize=True),
"Full":Y.value_counts(normalize=True)})

compare_samples

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)



import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.optimizers import Adam
from keras import backend as K


n_cols = X.shape[1]

def encoder_fx1(n_hidden_nodes_1=20,n_hidden_nodes_2=10, dropout_p = 0.5):

    input = Input(shape=(n_cols, ))
    encode_1 = Dense(n_hidden_nodes_1, input_shape = (n_cols,), activation="relu")(input)
    drop = Dropout(dropout_p, seed=1234)(encode_1)
    bn = BatchNormalization()(drop)

    encoded = Dense(n_hidden_nodes_2, input_shape = (n_cols,), activation="relu")(bn)
    drop = Dropout(dropout_p, seed=1234)(encoded)
    bn = BatchNormalization()(drop)

    decoded = Dense(n_hidden_nodes, input_shape = (n_cols,), activation="relu")(bn)

    model = Model(input, decoded)

    return(model)

def encoder_fx2(n_hidden_nodes_1 = 10, dropout_p = 0.3):

    input = Input(shape=(n_cols, ))
    encode_1 = Dense(n_hidden_nodes_1, activation="relu")(input)
    bn = BatchNormalization()(encode_1)
    drop = Dropout(dropout_p, seed=1234)(bn)

    decoded = Dense(n_cols, activation="relu")(drop)
    model = Model(input, decoded)

    return(model)

def col_pct_diff(y_true, y_pred):
    diff = (y_true, y_pred)
    pct = diff/y_true
    avg_diff = K.mean(pct)
    return(avg_diff)




encoder = encoder_fx2(60)
encoder.compile(optimizer="adam", loss="binary_crossentropy", metrics = [col_pct_diff])
# encoder.compile(optimizer="adam", loss="mse")




encoder.summary()

model1 = encoder.fit(X_train, X_train, epochs = 11, validation_split=0.1)


#################
model_test = encoder_fx2()
encoder.compile(optimizer="adam", loss="mse", metrics =[col_pct_diff] )

dummy_data = X_train.iloc[:100, :]
encoder.fit(dummy_data, dummy_data, epochs = 1)


#################


model1 = encoder.fit(X_train, X_train, epochs = 15, validation_split=0.1)
K.get_value(encoder.optimizer.lr)
encoder.optimizer.lr = 0.0001

model2 = encoder.fit(X_train, X_train, epochs = 15, validation_split=0.1)

# Model evaulation
model1.history.keys()
from plot_nn_metric import plot_nn_metric
plot_nn_metric(model1.history)
plot_nn_metric(model2.history)

encoder.summary()

#####

train_preds = encoder.predict(X_train)
train_preds.shape == X_train.shape

diff_dict = {}
for col_index in range(X_train.shape[1]):
    col_name = X_train.columns[col_index]
    diff = (X_train.iloc[:,col_index].astype(np.float64) - train_preds[:,col_index])/X_train.iloc[:,col_index].astype(np.float64)
    diff = [0 if x == -np.inf else x for x in diff]
    avg_diff = np.nanmean(diff)
    diff_dict[col_name] = avg_diff

train_diff = pd.Series(diff_dict)
#train_diff.head()
train_diff.describe()



##### Test set

preds = encoder.predict(X_test)
X_test.shape == preds.shape


diff_dict = {}
for col_index in range(X_test.shape[1]):
    col_name = X_test.columns[col_index]
    diff = (X_test.iloc[:,col_index].astype(np.float64) - preds[:,col_index])/X_test.iloc[:,col_index].astype(np.float64)
    diff = [0 if x == -np.inf else x for x in diff]
    avg_diff = np.nanmean(diff)
    diff_dict[col_name] = avg_diff

diff = pd.Series(diff_dict)
diff.head()
diff.describe()
