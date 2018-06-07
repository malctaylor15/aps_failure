# Functions for tree based methods

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def get_cum_feat_imp(feat_imp):

    """
    Get cumulative feature importance and differential feature importance
    """

    # Make sure Feat_Importance is a column
    col_str = ""
    for col in feat_imp.columns: col_str = col_str+", "+col
    error_msg = "Feat_Importance not in columns. Cols of df: "+col_str
    if "Feat_Importance" not in feat_imp.columns: raise KeyError(error_msg)

    # Add cumulative importance and difference to data frame
    feat_imp = feat_imp.sort_values(ascending = False, by = "Feat_Importance")
    feat_imp["Cum_Imp"] = feat_imp["Feat_Importance"].cumsum()
    feat_imp["Cum_Imp"] = feat_imp["Cum_Imp"].fillna(method="bfill")
    feat_imp["Diff_Feat_Imp"] = feat_imp["Feat_Importance"].diff()* -1
    feat_imp["Diff_Feat_Imp"] = feat_imp["Diff_Feat_Imp"].fillna(method="bfill")
    feat_imp["Diff_Feat_Imp"] = MinMaxScaler().fit_transform(feat_imp["Diff_Feat_Imp"].values.reshape(-1,1))
    return (feat_imp)

def plot_cum_feat_imp(feat_imp, title = "Cumulative Feature Importance"):


    numb_feat_range = range(0,len(feat_imp))

    plt.ioff()
    plt.title(title)
    line1, = plt.plot(numb_feat_range, feat_imp["Cum_Imp"], color = "blue", label = "Cumulative Importance")
    line2, = plt.plot(numb_feat_range, feat_imp["Diff_Feat_Imp"], color = "red", label = "Differential Feature Importance")
    plt.ylabel("Cumulative Feature importance")
    plt.xlabel("Number of variables")
    # plt.xticks(np.arange(0, len(feat_imp), 10))
    plt.legend([line1, line2])

    plt.show()
    plt.close()

def custom_score_aps(y_true, y_preds, pct=True, show_output = True):
    conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_preds), \
    columns = ["Pred_0", "Pred_1"], index = ["Actual_0", "Actual_1"])

    if show_output: print(conf_matrix)

    weighted_score = 10 * conf_matrix["Pred_0"].loc["Actual_1"] \
    + 500 * conf_matrix["Pred_1"].loc["Actual_0"]

    if pct == True:
        weighted_score = weighted_score/len(y_true)

    return(weighted_score)
