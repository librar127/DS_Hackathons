
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


def save_prediction_to_csv(y_pred):
    """
    Use this function to save your prediction result to a csv file.
    The resulting csv file is named as [team_name].csv

    :param y_pred: an array or a pandas series that follows the SAME index order as in the testing data
    """
    pd.DataFrame(dict(
        target=y_pred
    )).to_csv('predictions.csv', index=False, header=False)
 
 
 
 #### Read Train and Test Data
training_data = pd.read_csv('/home/rajneesh/Desktop/Queen_City_Hackathon/training.csv', index_col=0)
testing_data = pd.read_csv('/home/rajneesh/Desktop/Queen_City_Hackathon/testing.csv', index_col=0)

### ##############################################################
# Create correlation matrix
corr_matrix = training_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
columns_to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
if 'target' in columns_to_drop:
    columns_to_drop.remove('target')

### Drop all the co-related columns
data_nodups_df = training_data.drop(columns=columns_to_drop)
### ##############################################################

#### Data Preparation
data_nonull_df = data_nodups_df.fillna(data_nodups_df.mean())

y = data_nonull_df['target']
X = data_nonull_df.drop(columns=['target'])

### Create Data Matrix for fast computation
data_dmatrix = xgb.DMatrix(data=X, label=y)

### ##############################################################
#### Train very first model - XGB Regressor
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1, 'max_depth': 3, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5, num_boost_round=1000, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

### Train the model
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=1000)

### ##############################################################
#### Drop same column which has been dropped in training set
test_nodups_df = testing_data.drop(columns=columns_to_drop)
#### Fill missing values by the values which are used during the training data
test_nodups_df = test_nodups_df.fillna(data_nodups_df.mean())

### Convert into DMAtrix and predict
predictions = xg_reg.predict(xgb.DMatrix(data=test_nodups_df))
print(predictions.sum())
### ##############################################################
#save_prediction_to_csv(predictions)

