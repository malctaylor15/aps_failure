# Air Pressure System Component Failure Analysis

In this set of analysis, we examine data from heavy Scania trucks in heavy usage. The straight forward analysis is to model the air pressure system failure with associated costs to incorrectly classifying the failures and non failures. The problem has a strong class imbalance as failures are rare.

In this directory, there are two sets of analysis. One set of scripts are related to using Locally interpretable model explanations (LIME) to explore model predictions and the other explores using an auto encoder to reduce the dimensionality of the data.


## About the Data

The dataset was originally found on UCI Machine Learning Repository [here](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks) .

The data comes from the a fleet of Scania trucks in heavy usage. We are modeling the failure of components in the air pressure system. Unfortunately the meaning of the variables was not available due to proprietary reasons.

From the data description "
"

The dataset consists of data collected from heavy Scania
trucks in everyday usage. The system in focus is the
Air Pressure system (APS) which generates pressurised
air that are utilized in various functions in a truck,
such as braking and gear changes. The datasets'
positive class consists of component failures
for a specific component of the APS system.
The negative class consists of trucks with failures
for components not related to the APS. The data consists
of a subset of all available data, selected by experts.
"


When doing a classification analysis, there are different costs associated with incorrect predictions.

From the data description "


Cost-metric of miss-classification:

Predicted class | True class |
| pos | neg |
-----------------------------------------
pos | - | Cost_1 |
-----------------------------------------
neg | Cost_2 | - |
-----------------------------------------
Cost_1 = 10 and cost_2 = 500

The total cost of a prediction model the sum of 'Cost_1'
multiplied by the number of Instances with type 1 failure
and 'Cost_2' with the number of instances with type 2 failure,
resulting in a 'Total_cost'.

In this case Cost_1 refers to the cost that an unnessecary
check needs to be done by an mechanic at an workshop, while
Cost_2 refer to the cost of missing a faulty truck,
which may cause a breakdown.

Total_cost = Cost_1*No_Instances + Cost_2*No_Instances.  
"

THe dataset and full description can be found [here](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)


### Autoencoder Analysis

There are several files in the auto encoder analysis folder.

The main script is the autoencoder_main.py which has the workflow of gathering the data, cleaning, rescaling, trianing the model and evaluation of the model.

autoencoders_model_fx.py has the autoencoder model architecture.

plot_nn_metric.py has a function for plotting evaluation metrics over epoch for multiple data sets (train and validation).


### Autoencoder Discussion

One of the concerns I had when building the model was an appropriate evaluation metric. I wanted to make sure the final predictions were correctly associated with the inputs. I wanted to know how different and how many observations were from the original value.

One step that I decided to take was to cap and floor the predictions for each variable at 0 and 1.0 respectively. As part of the data preparation, I used a MinMaxScaler such that the maximum value of the training set was 0 and 1. I used the same scaler that was "trained" on the train set to scale the test set.

I decided to use hard cutoff offs of the percentage difference and then calculate the percentage of each variable that was past that cutoff.


## LIME Analysis

Locally interpretable model explanations is a way to understand model predictions. In a highly simplified terms, the idea is to preturb the input variables for a single observation and model the changes to the score with the changes in the input. Using this approach, we can understand any model in a similar way and create a uniform metric for understanding how the input variables affect the score. For more details on the theory, the paper can be found [here](https://arxiv.org/abs/1602.04938)


In this analysis, I conduct several tests using LIME.  I use a random forest as the model and look at the standard variable importance to get a better understanding of how the model is performing.

Next I try to understand which variables affect different ranked tiles. I was looking to see if the higher scoring observations were affected by a different variable than the lower tiles. This can be compared with the overall model variable importances. 
