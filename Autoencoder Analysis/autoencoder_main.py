import pandas as pd
import numpy as np
import sys
import os

# Add Autoencoder directory to path... later will need to add auto encoder functional files
sys.path.append(os.getcwd()+"/Autoencoder Analysis")
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
encoder.optimizer.lr = 0.001
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

within_match = abs(X_train.astype(np.float64) - train_preds)
#within_match2.apply(lambda x: sum(True if row>0.05 else False for row in x), axis = 0)

# Check the match differences less than x
# Want the counts to be high (so that means more observations are close to the originals)
numb_match = within_match.apply(lambda x: sum(x.le(0.05))/len(x))
numb_match.head()
numb_match.describe()

numb_match2 = within_match.apply(lambda x: sum(x.le(0.15))/len(x))
numb_match2.head()
numb_match2.describe()

# Number of observations that have less than 6000 matches within the limit
len(numb_match[numb_match <0.80])
numb_match[numb_match <0.80]


##### Test set

test_preds = encoder.predict(X_test)
X_test.shape == test_preds.shape

test_diff = X_test.astype(np.float64) - test_preds
test_numb_match = test_diff.apply(lambda x: sum(x.le(0.15))/len(x))
test_numb_match.head()
test_numb_match.describe()

len(test_numb_match[test_numb_match <0.60])
test_numb_match[test_numb_match <0.60]



# One flaw of this approach is that variables in the test set may  have scaled values greater than 1.0. If there are variables with values greater than 1.0, the cutoff would prevent the model from correctly identifying those values.  
