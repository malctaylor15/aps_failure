# Exmaine autoencoded values

import pandas as pd
import numpy as np
import os
import sys

data = pd.read_csv("Autoencoder Analysis/Encodings_with_original.csv")
data.shape
data.columns[58:63]
data.iloc[:3,:5]

embeddings = data.iloc[:,:59]
dep_var = data["class"]
original_data = data.iloc[:,61:]

# Understand more about the embeddings
# Want to make sure there is enough variation before comparing analysis with original data
embeddings.nunique()
embeddings.describe()
