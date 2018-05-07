import pandas as pd
import numpy as np

#Data from https://archive.ics.uci.edu/ml/machine-learning-databases/00421/

#data_raw = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv", skiprows = 20)
data_raw = pd.read_csv("aps_failure_training_set.csv", skiprows = 20)

# Try to understand data
data_raw.shape
data_raw.head()

data_raw_small  = data_raw.iloc[0:10000,:]

#data_raw_small.apply(pd.value_counts)["na"]

from clean_data1 import clean_data
X, Y = clean_data(data_raw_small)

Y.value_counts(normalize=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state = 1)

compare_samples = pd.DataFrame({"Train":y_train.value_counts(normalize=True),
"Test": y_test.value_counts(normalize=True),
"Full":Y.value_counts(normalize=True)})

compare_samples

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

####
max_train = X_train.apply(np.max, axis = 0)
max_test = X_test.apply(np.max, axis = 0)
len(max_train)
len([col_max for col_max in max_test if col_max > 1])
(X_train == 0).sum(axis=0)
(data_raw == "").sum(axis=0)



import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.optimizers import Adam
from keras import backend as K

import autoencoders_model_fx
from importlib import reload
reload(autoencoders_model_fx)
from autoencoders_model_fx import encoder_fx1, encoder_fx2, col_pct_diff
# from autoencoders_model_fx import *


encoder = encoder_fx2(60)
encoder.compile(optimizer="adam", loss="binary_crossentropy")#, metrics = [col_pct_diff])
# encoder.compile(optimizer="adam", loss="mse")

encoder.summary()
K.get_value(encoder.optimizer.lr)
encoder.optimizer.lr = 0.00005
model1 = encoder.fit(X_train, X_train, epochs = 10, validation_split=0.1)


# Model evaulation
model1.history.keys()
from plot_nn_metric import plot_nn_metric
plot_nn_metric(model1.history)


#####

train_preds = encoder.predict(X_train)
train_preds = pd.DataFrame(np.clip(train_preds, 0, 1), columns = X_train.columns)
train_preds.shape == X_train.shape

train_preds.iloc[0:5, 0:5]
X_train.iloc[0:5, 0:5]


[x for x in train_preds.isnull().sum(axis = 0) if x >0 ]


# How many predictions are within 10% of train

within_match = abs(X_train.iloc[:,col_index].astype(np.float64) - train_preds.iloc[:,col_index])
(within_match > 0).sum()

within_match2 = abs(X_train.astype(np.float64) - train_preds)
np.sort((within_match2 > 0.05).sum(axis= 0))
np.mean((within_match2 > 0.05).sum(axis= 0))



diff_dict = {}
for col_index in range(X_train.shape[1]):
    col_name = X_train.columns[col_index]
    #diff = (X_train.iloc[:,col_index].astype(np.float64) -
    #train_preds.iloc[:,col_index])(X_test.iloc[:,col_index].astype(np.float64) - preds[:,col_index/X_train.iloc[:,col_index].astype(np.float64)

    diff = [0 if x == -np.inf else x for x in diff]
    diff = np.nanmean(diff)
    diff_dict[col_name] = diff

train_diff = pd.Series(diff_dict)
train_diff.head()
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
