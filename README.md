# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE

```
from sklearn.datasets import load_boston
boston_data=load_boston()
import pandas as pd
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV
boston.head(10)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from math import sqrt

cv = KFold(n_splits=10, random_state=None, shuffle=False)
classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))

boston.var()

X = X.drop(columns = ['NOX','CHAS'])
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))

# Filter Features by Correlation
import seaborn as sn
import matplotlib.pyplot as plt
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sn.heatmap(boston.corr(), ax=ax)
plt.show()
abs(boston.corr()["MEDV"])
abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>0.5].drop('MEDV')).index.tolist()
vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
for val in vals:
    features = abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>val].drop('MEDV')).index.tolist()
    
    X = boston.drop(columns='MEDV')
    X=X[features]
    
    print(features)

    y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
    print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
    print("R_squared: " + str(round(r2_score(y,y_pred),2)))

# Feature Selection Using a Wrapper

boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
boston['RAD'] = boston['RAD'].astype('category')
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(classifier_pipeline, 
           k_features=1, 
           forward=False, 
           scoring='neg_mean_squared_error',
           cv=cv)

X = boston.drop(columns='MEDV')
sfs1.fit(X,y)
sfs1.subsets_

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']].corr()

boston['RM*LSTAT']=boston['RM']*boston['LSTAT']

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

sn.pairplot(boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']])

boston = boston.drop(boston[boston['MEDV']==boston['MEDV'].max()].index.tolist())

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT','RM*LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston['LSTAT_2']=boston['LSTAT']**2

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))
```
#OUTPUT :

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/4a103f60-6868-49f5-ab32-49b21d2da2af)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/61ebb542-cb51-4989-9e9c-1d9ca736c59d)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/d9a61c14-8017-4bf1-b74b-26f82b2d1923)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/0f01bec6-1b68-49df-9061-51e20fcac0d1)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/7d251524-e69d-4992-95fa-48b837693789)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/9f5b72a3-3aaa-4d01-90a3-1bbae24773b5)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/9c8bd47c-cd68-4863-841f-3c9a0cf36f91)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/185b08f6-e4ab-444a-9f4f-7e7e73b19ce7)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/7c90d7f3-f412-4793-ae74-26a3e6f5ea91)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/d6b2c944-e7a8-47a6-8f8f-36b962adf7de)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/10daf2b9-0757-425c-b2f2-4c64d19f41fa)

![image](https://github.com/NivethaKumar30/Ex-07-Feature-Selection/assets/119559844/e88d01f4-b2ca-4d6a-9c05-a6c9c1c9d7d1)


RESULT:

Various feature selection techniques have been performed on a given dataset successfully.
