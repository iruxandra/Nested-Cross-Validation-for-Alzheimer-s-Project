# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 04:01:07 2020

@author: Ruxandra Ion
"""

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
df2_removed_suvr = df2[df2.columns[~df2.columns.isin(['VENTRICLE_5TH_SUVR'])]]
df2_removed= df2_removed_suvr[df2_removed_suvr.columns[~df2_removed_suvr.columns.isin(['VENTRICLE_5TH_VOLUME'])]]


#Merging the data sets by VISCODE and RID
results = pd.merge(df1, df2_removed, left_on=['RID', 'VISCODE'], right_on=['RID','VISCODE2'])
results['Age_at_scan'] = results.apply(lambda x: x['AGE'] + x['Years_bl'], axis = 1)


#this selects all MCI, RID are repeated - need to change to select the first occuring MCI diagnostic for one RID.
test_mci = pd.DataFrame(results, columns = ['RID', 'DX'])
#generates a column : age_at_scan which is AGE + Years_bl
#get RIDs correspondent only to DX['MCI']
test_mciresults = test_mci.groupby(['DX']).get_group('MCI') 

#Merge min (baseline) with full data frame to get a subset
results_mci = pd.merge(results, test_mciresults) #contains all MCI patients

#must select one RID per MCI diagnostic
##This selects the first scan for all patients based on the minimum age at PET Scan
test = pd.DataFrame(results, columns = ['RID', 'Age_at_scan'])
#generates a column : age_at_scan which is AGE + Years_bl
test_minresults = test.groupby(['RID'], as_index = False).min() #shows baseline measurements
#Merge min (baseline) with full data frame to get a subset

results_bl_mci = pd.merge(results_mci, test_minresults) #contains all VOLUMETRIC and SUVR data for baseline
results_bl_mcitest = results_bl_mci.groupby(['Age_at_scan'], as_index = False).min()
#Assigning values to variables
X = results_bl_mcitest.loc [:,'BRAAK56_VOLUME']
ICV_dummy = results_bl_mcitest.loc[:, 'ICV_bl']
ICV_dummy.fillna(value=np.mean(ICV_dummy), inplace = True) #handling missing data
X_norm = X.apply(lambda x: x/ICV_dummy) #axis=0)
#normalizing by the recommended reference region (white matter)
y_norm = results_bl_mcitest.loc [:,'BRAAK56_SUVR']/ results_bl_mcitest.loc[:, 'ERODED_SUBCORTICALWM_SUVR']  


#Splitting the data set intro Training Set and Testing Set
from sklearn.model_selection import train_test_split
X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X_norm, y_norm, test_size = 0.2)
#the training set contains a known output and the model learns on this data                  
#the test set tests our model's predictions based on the training set

#Feature Scailing for accurate predictions
#feature scaling helps normalise the data within particular range, handling outliers

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

lin_ref_mciA = LinearRegression(normalize=True)
lin_ref_mciA.fit(X_norm_train, y_norm_train)
y_ref_pred=lin_ref_mciA.predict(X_norm_test)
MSE_ref_mciA = mean_squared_error(y_norm_test,y_ref_pred)
print(MSE_ref_mciA)

#Correlation Coefficient
corr_coef_ref_mciA,_ = pearsonr(y_norm_test, y_ref_pred)
print('Pearsons correlation: %.3f' % corr_coef_ref_mciA)
#r=0.552


fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_test, y_ref_pred, color = 'brown')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Scatter Plot Test Set (MCI BRAAK A)')
pyplot.show()

##Braak B
lin_ref_mciB = LinearRegression(normalize=True)
lin_ref_mciB.fit(X_norm_train, y_norm_train)
y_ref_pred=lin_ref_mciB.predict(X_norm_test)
MSE_ref_mciB = mean_squared_error(y_norm_test,y_ref_pred)
print(MSE_ref_mciB)

#Correlation Coefficient
corr_coef_ref_B,_ = pearsonr(y_norm_test, y_ref_pred)
print('Pearsons correlation: %.3f' % corr_coef_ref_B)
#r=0.552


fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_test, y_ref_pred, color = 'brown')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Scatter Plot Test Set (MCI BRAAK B)')
pyplot.show()

##Braak C

lin_ref_mciC = LinearRegression(normalize=True)
lin_ref_mciC.fit(X_norm_train, y_norm_train)
y_ref_pred=lin_ref_mciC.predict(X_norm_test)
MSE_ref_mciC = mean_squared_error(y_norm_test,y_ref_pred)
print(MSE_ref_mciC)

#Correlation Coefficient
corr_coef_ref_mciC,_ = pearsonr(y_norm_test, y_ref_pred)
print('Pearsons correlation: %.3f' % corr_coef_ref_mciC)
#r=0.552


fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_test, y_ref_pred, color = 'brown')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Scatter Plot Test Set (MCI BRAAK C)')
pyplot.show()

