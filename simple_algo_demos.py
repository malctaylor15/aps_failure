import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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

estimator = RandomForestClassifier(n_estimators = 40, max_depth = 4, random_state=4, class_weight={1:1, 0:50})
estimator.fit(X_train, y_train)

estimator.score(X_train, y_train)
preds1 = estimator.predict(X_train)
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_train, preds1), columns = ["Actual_0", "Actual_1"], index = ["Pred_0", "Pred_1"])

# Test set evaluation

preds_test = estimator.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, preds_test), columns = ["Actual_0", "Actual_1"], index = ["Pred_0", "Pred_1"])

len(estimator.feature_importances_)

# Feature importance stuff
feat_imp = pd.DataFrame(estimator.feature_importances_, index = X_train.columns, columns = ["Feat_Importance"])
feat_imp.sort_values(ascending = False, by = "Feat_Importance", inplace=True)
feat_imp.head()
feat_imp.tail()
sum(feat_imp.iloc[:,0] == 0)
feat_imp.sum(axis=0)

feat_imp["Cum_Imp"] = feat_imp["Feat_Importance"].cumsum()
feat_imp["Diff_Feat_Imp"] = feat_imp["Feat_Importance"].diff()
(feat_imp["Diff_Feat_Imp"]*-40).head()

numb_feat_range = range(0,len(feat_imp))

plt.ioff()
plt.title("Cumulative Feature Importance")
line1, = plt.plot(numb_feat_range, feat_imp["Cum_Imp"], color = "blue", label = "Cumulative Importance")
line2, = plt.plot(numb_feat_range, -40*feat_imp["Diff_Feat_Imp"], color = "red", label = "Differential Feature Importance")
plt.ylabel("Cumulative Feature importance")
plt.xlabel("Number of variables")
# plt.xticks(np.arange(0, len(feat_imp), 10))
plt.legend([line1, line2])

plt.show()

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
xgb_dtrain = xgb.DMatrix(X_train.values, label = y_train, weight = set_weights, feature_names = X_train.columns)
xgb_dtest = xgb.DMatrix(X_test.values, label = y_test, feature_names = X_test.columns)

# Parameters and train model
params = {'max_depth':3, 'eta':0.3, 'silent':1, 'objective':'binary:logistic' }
n_rounds = 40
xgb_model = xgb.train(params, xgb_dtrain, n_rounds)

### Begin model analysis
xg_feat_imp = xgb_model.get_fscore()
xg_feat_imp_pd = pd.DataFrame(xgb_model.get_fscore() , index = ["Feat_Importance_Raw"]).T
xg_feat_imp_pd.columns
xg_feat_imp_pd.sort_values(ascending = False, by="Feat_Importance_Raw", inplace= True)
xg_feat_imp_pd.head()
# Create dataframe with columns to RF above
xg_feat_imp_pd["Feat_Importance"] = xg_feat_imp_pd["Feat_Importance_Raw"]/xg_feat_imp_pd["Feat_Importance_Raw"].sum()
xg_feat_imp_pd["Cum_Imp"] = xg_feat_imp_pd["Feat_Importance"].cumsum()
xg_feat_imp_pd["Diff_Feat_Imp"] = xg_feat_imp_pd["Feat_Importance"].diff()
xg_feat_imp_pd.head(10)

# Create similar plot as to above
numb_feat_range = range(0,len(xg_feat_imp_pd))
plt.ioff()
plt.title("Cumulative Feature Importance")
line1, = plt.plot(numb_feat_range, xg_feat_imp_pd["Cum_Imp"], color = "blue", label = "Cumulative Importance")
line2, = plt.plot(numb_feat_range, -40*xg_feat_imp_pd["Diff_Feat_Imp"], color = "red", label = "Differential Feature Importance")
plt.ylabel("Cumulative Feature importance")
plt.xlabel("Number of variables")
# plt.xticks(np.arange(0, len(feat_imp), 10))
plt.legend([line1, line2])
plt.show()






######## Compare feature importances
xg_feat_imp_pd.head()

feat_imp.head()


# Feature ordering -- see if RF matches GB (gb truth)
pd.merge ....
# can compare relative feature importances
