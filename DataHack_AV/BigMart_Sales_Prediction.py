'''
Created on Feb 2, 2016

@author: rajneesh_kumar
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("E:\\Python\\AV_Datahack\\Datasets\\BigMart\\BigMart_Train.csv")
#print df.apply(lambda x: sum(x.isnull()), axis =0)

def prepare_data(df):
    
    df['Outlet_Size'].fillna('Medium',inplace=True)
    df['Item_Weight'].fillna(df['Item_Weight'].dropna().mean(), inplace=True)
    #print df.apply(lambda x: sum(x.isnull()), axis =0)
       
    Outlet_Size_mappings = {'High':3,
                            'Medium':2,
                            'Small':1}    
    df['Outlet_Size'] = df['Outlet_Size'].map(Outlet_Size_mappings)
    
    Outlet_Location_mappings = {'Tier 3':1,
                                'Tier 2':2,
                                'Tier 1':3}
    df['Outlet_Location_Type'] = df['Outlet_Location_Type'].map(Outlet_Location_mappings)
    
    df = pd.concat([
                    df['Item_Weight'],
                    #pd.get_dummies(df['Item_Fat_Content']),
                    df['Item_Visibility'],                    
                    #pd.get_dummies(df['Item_Type'], prefix = "Item_Type"),
                    df['Item_MRP'],
                    #df['Outlet_Size'],
                    #df['Outlet_Location_Type'],
                    pd.get_dummies(df['Outlet_Type'])
                    ], axis =1)
    

    #print df.apply(lambda x: sum(x.isnull()), axis =0)
    return df

headers = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 
           'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

'''
df = df[df.Item_Fat_Content != 'LF']
df = df[df.Item_Fat_Content != 'reg']
df = df[df.Item_Fat_Content != 'low fat']
'''

X = prepare_data(df[headers])
y = df['Item_Outlet_Sales'] 

#model = LinearRegression()
model = GradientBoostingRegressor(n_estimators=500,  max_depth=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(X_train, y_train)
#print(model.coef_)
print model.score(X, y) 
print "Residual sum of squares: %.2f" % np.mean((model.predict(X_test) - y_test) ** 2)

test_data = pd.read_csv("E:\\Python\\AV_Datahack\\Datasets\\BigMart\\BigMart_Test.csv")
XX = prepare_data(test_data[headers])

output_df = pd.DataFrame({
        "Item_Identifier": test_data['Item_Identifier'],
        "Outlet_Identifier": test_data['Outlet_Identifier'],
        "Item_Outlet_Sales":model.predict(XX)
})
output_df.to_csv('E:\\Python\\AV_Datahack\\Datasets\\BigMart\\BigMart_Test_Op.csv', index=False)   
