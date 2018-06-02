import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from importlib import reload

import tree_utils
reload(tree_utils)
from tree_utils import *

%matplotlib inline
!pwd
os.chdir("/home/malcolm/Documents/aps_failure")

data_raw = pd.read_csv("aps_failure_training_set.csv", skiprows = 20)

# Try to understand data
data_raw.shape
data_raw.head()

data_raw_small  = data_raw.iloc[0:10000,:]

#data_raw_small.apply(pd.value_counts)["na"]

from clean_data1 import clean_data
X, Y = clean_data(data_raw_small)

Y.value_counts(normalize=True)

# Y = Y.apply(lambda x: 1 if x =="neg" else 0)
Y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state = 1)

compare_samples = pd.DataFrame({"Train":y_train.value_counts(normalize=True),
"Test": y_test.value_counts(normalize=True),
"Full":Y.value_counts(normalize=True)})

compare_samples

1/compare_samples["Train"].loc[0]



from sklearn.ensemble import RandomForestClassifier


estimator = RandomForestClassifier(n_estimators = 40, max_depth = 4, random_state=4, class_weight={1:10, 0:500})
estimator.fit(X_train, y_train)

estimator.score(X_train, y_train)
preds1 = estimator.predict(X_train)

from sklearn.metrics import confusion_matrix
custom_score_aps(y_train, preds1)
pd.DataFrame(confusion_matrix(y_train, preds1), columns = ["Actual_0", "Actual_1"], index = ["Pred_0", "Pred_1"])

# Test set evaluation

preds_test = estimator.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, preds_test), columns = ["Actual_0", "Actual_1"], index = ["Pred_0", "Pred_1"])

len(estimator.feature_importances_)

# Feature importance stuff
feat_imp = pd.DataFrame(estimator.feature_importances_, index = X_train.columns, columns = ["Feat_Importance"])

feat_imp2 = get_cum_feat_imp(feat_imp)
plot_cum_feat_imp(feat_imp2)
custom_score_aps(y_test, preds_test)


cum_cutoff = 0.90
indiv_cutoff = 0.002
high_imp_feats1 = feat_imp[feat_imp["Cum_Imp"] < cum_cutoff]
len(high_imp_feats1)
high_imp_feats2 = feat_imp[feat_imp["Feat_Importance"] > indiv_cutoff]
len(high_imp_feats2)



####################
##### XGBOOST ######
####################


from xgboost import XGBClassifier
import xgboost as xgb

# Set weights for high and low class and set up data frames for xgb
set_weights_tr = [1 if y == 1 else 100 for y in y_train]
xgb_dtrain = xgb.DMatrix(X_train.values, label = y_train, weight = set_weights_tr, feature_names = X_train.columns)
xgb_dtest = xgb.DMatrix(X_test.values, label = y_test, feature_names = X_test.columns)

# Parameters and train model
params = {'max_depth':4, 'eta':0.2, 'silent':1, 'objective':'binary:logistic' }
n_rounds = 100
xgb_model = xgb.train(params, xgb_dtrain, n_rounds)

# Get predictions
import tree_utils
reload(tree_utils)
from tree_utils import *

xg_preds_test = xgb_model.predict(xgb_dtest)

### Begin model analysis
xg_feat_imp = xgb_model.get_fscore()
xg_feat_imp_pd = pd.DataFrame(xgb_model.get_fscore() , index = ["Feat_Importance_Raw"]).T
xg_feat_imp_pd["Feat_Importance"] = xg_feat_imp_pd["Feat_Importance_Raw"]/xg_feat_imp_pd["Feat_Importance_Raw"].sum()
xg_feat_imp_pd.sort_values(ascending = False, by="Feat_Importance_Raw", inplace= True)

xg_feat_imp2 = get_cum_feat_imp(xg_feat_imp_pd)
plot_cum_feat_imp(xg_feat_imp2)
custom_score_aps()




######## Compare feature importances
xg_feat_imp_pd.head()
# Feature ordering -- see if RF matches GB (gb truth)
pd.merge()
# can compare relative feature importances
