
'''
Created on Jan 7, 2016

@author: rajneesh_kumar
'''
import pandas as pd
import time
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

s_time = int(round(time.time()))
train_file = "Kaggle_Titanic_Train_Original.csv"
test_file = "Kaggle_Titanic_Test.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
y = train_data['Survived']

def prepare_data(data):
    
    X = pd.concat([data[['Fare', 'Pclass', 'Age']], 
                           pd.get_dummies(data['Sex'], prefix="Sex"), 
                           pd.get_dummies(data['Embarked'], prefix="Embarked"),
                           data['Parch'],
                           data['SibSp']], axis=1)
    
    median_features = X.dropna().median()
    imputed_features = X.fillna(median_features)
    
    print imputed_features.count()
    return imputed_features

'''
print "\nDivide the data into train and test"
X_train, X_test, y_train, y_test = train_test_split(imputed_features, y, test_size =0.20, random_state=0)
print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape
print y_train.value_counts()
print y_test.value_counts()
'''

print "\n#######################################################################"
train_features = prepare_data(train_data)

print "\n#######################################################################"
rfc = RandomForestClassifier(
      random_state=1,
      n_estimators=250,
      min_samples_split=4,
      min_samples_leaf=2
)
abc = AdaBoostClassifier(rfc ,n_estimators=20,  learning_rate=1.0, algorithm='SAMME.R')
scores = cross_val_score(abc, train_features, y, cv=10 )

print "Random Forest CV scores:"
print "Accuracy## min: {0:.2f}, mean: {1:.2f}, max: {2:.2f}".format(scores.min(), scores.mean(), scores.max())

print "######################## Working with Test Data #########################"

rfc.fit(train_features, y)
passengerid = test_data.PassengerId
test_features = prepare_data(test_data)
prediction = rfc.predict(test_features)

output_df = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": prediction
})
output_df.to_csv('Titanic_Survival_Op.csv', index=False) 


e_time = int(round(time.time()))
print "Total processing Time in Seconds: ", (e_time-s_time)
