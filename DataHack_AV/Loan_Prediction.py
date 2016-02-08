'''
Created on Jan 7, 2016

@author: rajneesh_kumar
'''

import pandas as pd
from sklearn.cross_validation import cross_val_score
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

s_time = int(round(time.time()))

def prepare_data(original_data):
    
    
    original_data['TotalIncome'] = original_data['ApplicantIncome'] + original_data['CoapplicantIncome']
    original_data['TotalIncome_log'] = np.log(original_data['TotalIncome'])
    original_data['LoanAmount_log'] = np.log(original_data['LoanAmount'])
    
    one_hot_encoding_data = pd.concat([pd.get_dummies(original_data['Gender']),
                                       pd.get_dummies(original_data['Married'], prefix = "Married"),
                                       original_data['Dependents'],
                                       pd.get_dummies(original_data['Education'], prefix = "Education"),
                                       pd.get_dummies(original_data['Self_Employed'], prefix="Self_Employed"),
                                       original_data['TotalIncome_log'],#+original_data['CoapplicantIncome'],
                                       original_data['LoanAmount_log'],
                                       original_data['Loan_Amount_Term'],
                                       original_data['Credit_History'],
                                       pd.get_dummies(original_data['Property_Area'], prefix="Property_Area")                                       
                                       ], axis =1)
    
    #one_hot_encoding_data.drop('Female', 1, inplace=True)
    #one_hot_encoding_data.drop('Married_Yes', 1, inplace=True)
    #one_hot_encoding_data.drop('Education_Not Graduate', 1, inplace=True)    
    #one_hot_encoding_data.drop('Self_Employed_No', 1, inplace=True)
    
    one_hot_encoding_data[one_hot_encoding_data.Dependents == '3+'] = 5
    median_features = one_hot_encoding_data.dropna().median()    
    print median_features
    imputed_features = one_hot_encoding_data.fillna(median_features)
        
    #print imputed_features.head(1)
    #print imputed_features.count()
    
    return imputed_features

file_name = "E:\\Python\\AV_Datahack\\Datasets\\LoadPrediction\\DH_LoanPrediction_Train.csv"
data = pd.read_csv(file_name)

#print data.columns.values
headers = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome',
           'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

original_data = data[headers]
y = data['Loan_Status']

print "\n******************* Load Prediction Training *******************"
headers = ['Sex_Female', 'Sex_Male', 'Married_No', 'Married_Yes', 'Employed_No', 'Employed_Yes', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Rural',  'Property_Semiurban', 'Property_Urban' ]

    
original_data['LoanAmount'].fillna(original_data['LoanAmount'].mean(), inplace=True)
original_data['Self_Employed'].fillna('No',inplace=True)

X = prepare_data(original_data)

'''
le = LabelEncoder()
cl = le.fit_transform(y)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=1)

from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1, gamma=0.1)
'''

clf = RandomForestClassifier(
    random_state=1,
    n_estimators=500,
    min_samples_split=10,
    min_samples_leaf=10
)
#clf = LogisticRegression()

'''
model = LogisticRegression()
predictor_var = ['Male', 'Married_No', 'Education_Graduate',
       'Self_Employed_Yes', 'Loan_Amount_Term', 'Credit_History', 'Property_Area_Rural',
        'Property_Area_Semiurban', 'Property_Area_Urban', 'LoanAmount_log','TotalIncome_log']
scores = cross_val_score(model, X, y, cv=10)
print "Accuracy LR## min: {0:.2f}, mean: {1:.2f}, max: {2:.2f}".format(scores.min(), scores.mean(), scores.max())
'''

scores = cross_val_score(clf, X, y, cv=10)
print "Accuracy RF## min: {0:.2f}, mean: {1:.2f}, max: {2:.2f}".format(scores.min(), scores.mean(), scores.max())

print "\n******************* Load Prediction Testing *******************"
test_data = pd.read_csv("E:\\Python\\AV_Datahack\\Datasets\\LoadPrediction\\DH_LoanPrediction_Test.csv")
loan_ids = test_data['Loan_ID'].tolist()
#loan_ids.insert(0, "Loan_ID")

XX = prepare_data(test_data)
clf.fit(X, y)

'''
featimp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
sum_score = 0
for i in xrange(len(featimp)):
    sum_score += featimp[i]

print featimp[:50]
print sum_score
'''

loan_status = clf.predict(XX).tolist()
#loan_status.insert(0, "Loan_Status")

'''
dict_prediction = dict(zip(loan_ids, loan_status))
prediction_Op = pd.DataFrame.from_dict(dict_prediction, orient='index')

prediction_Op.to_csv("E:\\Python\\AV_Datahack\\Datasets\\LoadPrediction\\DH_LoanPrediction_Test_Op.csv")
print "File Written Successfully!!"
'''

output_df = pd.DataFrame({
        "Loan_ID": loan_ids,
        "Loan_Status": loan_status
})
output_df.to_csv('E:\\Python\\AV_Datahack\\Datasets\\LoadPrediction\\DH_LoanPrediction_Test_Op.csv', index=False)   
  

print "\n******************** Total Processing Time ********************"
e_time = int(round(time.time()))
print "Total Processing Time in Seconds: ", (e_time-s_time)
