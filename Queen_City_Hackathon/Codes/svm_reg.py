print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


# #############################################################################
training_data = pd.read_csv('/home/rajneesh/Desktop/Queen_City_Hackathon/training.csv', index_col=0)
testing_data = pd.read_csv('/home/rajneesh/Desktop/Queen_City_Hackathon/testing.csv', index_col=0)

# #############################################################################

# Create correlation matrix
corr_matrix = training_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
columns_to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]


### Drop all the co-related columns
data_nodups_df = training_data.drop(columns=columns_to_drop)
### ##############################################################

#### Data Preparation
data_nonull_df = data_nodups_df.fillna(data_nodups_df.mean())

y = data_nonull_df['target']
X = data_nonull_df.drop(columns=['target'])

### ##############################################################

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)

# #############################################################################
test_nodups_df = testing_data.drop(columns=columns_to_drop)
#### Fill missing values by the values which are used during the training data
test_nodups_df = test_nodups_df.fillna(data_nodups_df.mean())

# #############################################################################

kernel_label = ['linear', 'poly', 'rbf', 'sigmoid']

for each_kernel in kernel_label:
    model = SVR(kernel=each_kernel)
    model.fit(X, y)
    ### Convert into DMAtrix and predict
    predictions = model.predict(test_nodups_df)
    print(predictions.sum())

# #############################################################################
    
