import pandas as pd
import numpy as np

def clean_data(data_raw, pct_nas = 0.15):

    pct_na_dict = {col:data_raw[col].value_counts(normalize=True).loc["na"] for col in data_raw.columns if "na" in data_raw[col].value_counts().index}
    pct_na = pd.Series(pct_na_dict)
    large_pct_na = pct_na[pct_na > pct_nas].index
    data_raw.drop(large_pct_na, axis = 1, inplace= True)

    # Replace "na"s with the mean -- look into
    print("Beginning to drop na's and nan's... this may take a while ")
    data1 = data_raw.replace(["na", "nan"], [data_raw.mean(), data_raw.mean()])
    print("Finished replacing....")

    X = data1.drop(["class"], axis = 1)
    Y = data_raw["class"].apply(lambda x: 1 if x =="neg" else 0)
    

    print("Shape before: " + str(data_raw.shape))
    print("Shape after: "+ str(X.shape))

    return(X,Y)
