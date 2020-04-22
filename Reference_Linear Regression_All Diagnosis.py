# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 03:44:20 2020

@author: Ruxandra Ion
"""

##Reference model: Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#from sklearn.dataset import load_digits
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot
from sklearn.svm import SVR
import seaborn as sns;

#Merge data: ADNIMERGE with AV1415
dataset1 = pd.read_csv('ADNIMERGE.csv')
dataset2 = pd.read_csv('UCBERKELEYAV1451_08_27_19.csv')

df1 = pd.DataFrame(dataset1, columns = ['RID', 'VISCODE', 'DX_bl', 'DX',
                                        'AGE','PTGENDER','PTEDUCAT','PTRACCAT',
                                        'PTMARRY', 'Years_bl', 'ICV', 'ICV_bl'])


df2_test = pd.DataFrame(dataset2, columns = ['RID', 'VISCODE2'])
df2_autoSUVR = dataset2.loc [:, dataset2.columns.str.endswith('SUVR')]
df2_autoVOLUME = dataset2.loc [:, dataset2.columns.str.endswith('VOLUME')]


df2 = pd.concat( [df2_test, df2_autoSUVR, df2_autoVOLUME], axis = 1)
#handling missing data
df2_removed = df2[df2.columns[~df2.columns.isin(['VENTRICLE_5TH_SUVR'])]]

#Merging the data sets by VISCODE and RID
results = pd.merge(df1, df2_removed, left_on=['RID', 'VISCODE'], right_on=['RID','VISCODE2'])
results['Age_at_scan'] = results.apply(lambda x: x['AGE'] + x['Years_bl'], axis = 1)
test = pd.DataFrame(results, columns = ['RID', 'Age_at_scan'])

#selecting baseline measurements
test_minresults = test.groupby(['RID'], as_index = False).min()

#Merge min (baseline) with full data frame to get the subset of interest
results_bl = pd.merge(results, test_minresults) #contains all VOLUMETRIC and SUVR data for baseline

#Assigning values to variables
X = results_bl.loc [:,'BRAAK56_VOLUME']
ICV_dummy = results_bl.loc[:, 'ICV_bl']
ICV_dummy.fillna(value=np.mean(ICV_dummy), inplace = True) #handling missing data
X_norm = X.apply(lambda x: x/ICV_dummy) #axis=0)
#normalizing by the recommended reference region (white matter)
y_norm = results_bl.loc [:,'BRAAK56_SUVR']/ results_bl.loc[:, 'ERODED_SUBCORTICALWM_SUVR']  


#Splitting the data set intro Training Set and Testing Set
from sklearn.model_selection import train_test_split
X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X_norm, y_norm, test_size = 0.2)
#the training set contains a known output and the model learns on this data                  
#the test set tests our model's predictions based on the training set

#Feature Scailing for accurate predictions

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_norm_train = sc_X.fit_transform(X_norm_train)
X_norm_test = sc_X.transform(X_norm_test)
#means and strd dev applied here 
X_norm_train = pd.DataFrame(X_norm_train)
X_norm_train.fillna(value=0, inplace = True)
X_norm_test = pd.DataFrame(X_norm_test)
X_norm_test.fillna(value=0, inplace = True)

#Linear Regression Model: Reference point for performance evaluation
from sklearn.linear_model import LinearRegression

lin_ref_A = LinearRegression(normalize=True)
lin_ref_A.fit(X_norm_train, y_norm_train)
y_ref_pred=lin_ref_A.predict(X_norm_test)
MSE_ref_A = mean_squared_error(y_norm_test,y_ref_pred)
print(MSE_ref_A)

#Correlation Coefficient
corr_coef_ref_A,_ = pearsonr(y_norm_test, y_ref_pred)
print('Pearsons correlation: %.3f' % corr_coef_ref_A)
#r=0.552


fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_test, y_ref_pred, color = 'brown')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Scatter Plot Test Set (BRAAK A)')
pyplot.show()

##Braak B
lin_ref_B = LinearRegression(normalize=True)
lin_ref_B.fit(X_norm_train, y_norm_train)
y_ref_pred=lin_ref_B.predict(X_norm_test)
MSE_ref_B = mean_squared_error(y_norm_test,y_ref_pred)
print(MSE_ref_B)

#Correlation Coefficient
corr_coef_ref_B,_ = pearsonr(y_norm_test, y_ref_pred)
print('Pearsons correlation: %.3f' % corr_coef_ref_B)
#r=0.552


fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_test, y_ref_pred, color = 'brown')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Scatter Plot Test Set (BRAAK B)')
pyplot.show()

##Braak C

lin_ref_C = LinearRegression(normalize=True)
lin_ref_C.fit(X_norm_train, y_norm_train)
y_ref_pred=lin_ref_C.predict(X_norm_test)
MSE_ref_C = mean_squared_error(y_norm_test,y_ref_pred)
print(MSE_ref_C)

#Correlation Coefficient
corr_coef_ref_C,_ = pearsonr(y_norm_test, y_ref_pred)
print('Pearsons correlation: %.3f' % corr_coef_ref_C)
#r=0.552


fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_test, y_ref_pred, color = 'brown')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Scatter Plot Test Set (BRAAK C)')
pyplot.show()

