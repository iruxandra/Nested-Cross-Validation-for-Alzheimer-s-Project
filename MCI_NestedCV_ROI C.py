# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 01:24:19 2020

@author: Ruxandra Ion
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:08:49 2020

@author: Ruxandra Ion
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:53:34 2020

@author: Ruxandra Ion
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:35:01 2020

@author: Ruxandra Ion
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:34:31 2020

@author: Ruxandra Ion
"""


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

X = results_bl_mcitest.loc [:, results_mci.columns.str.endswith('VOLUME')] 
ICV_dummy = results_bl_mcitest.loc[:, 'ICV_bl']
ICV_dummy.fillna(value=np.mean(ICV_dummy), inplace = True)  
#results_bl.loc [:, 'ICV_bl']#all VOLUMETRIC info 
X_norm = X.apply(lambda x: x/ICV_dummy, axis=0) #all volumetric information
y_norm_mci = results_bl_mcitest.loc [:,'BRAAK56_SUVR']/ results_bl_mcitest.loc[:, 'ERODED_SUBCORTICALWM_SUVR']  # target variable #tau burden
#norm by white matter (?) check documentation

#test data set only with MCI
#training data - no one with ad, some with MCI, these should be CN.
#for scatter plot - color code 

#Splitting the data set intro Training Set and Test Set
from sklearn.model_selection import train_test_split
X_norm_train, X_norm_test, y_norm_mci_train, y_norm_mci_test = train_test_split(X_norm, y_norm_mci, test_size = 0.2) #random state only to recreate v similar results
#the training set contains a known output and the model learns on this data                  
#the test set tests our model's predictions

#Feature Scailing #for accurate predictions we will use this
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_norm_train = sc_X.fit_transform(X_norm_train)
X_norm_test = sc_X.transform(X_norm_test)
#feature scaling helps normalise the data with a particular range
#means and strd dev applied here 
X_norm_train = pd.DataFrame(X_norm_train)
X_norm_train.fillna(value=0, inplace = True)
#not sure why i get another NaN error here
X_norm_test = pd.DataFrame(X_norm_test)
X_norm_test.fillna(value=0, inplace = True)

#Testing Ridge Regression for MCI individuals
alphas =  np.logspace(start = -5, stop = 8, num = 80 ) #best param alpha
#alpha_bp = (0.8643882620598261)
print(alphas)
alphas
ridge_mci = Ridge(normalize = True)
coefs=[]
MSE = []
for a in alphas:
    ridge_mci.set_params(alpha=a)
    ridge_mci.fit(X_norm_train, y_norm_mci_train)
    pred_ridge_mci = ridge_mci.predict(X_norm_test)
    MSEtemp = mean_squared_error(y_norm_mci_test, pred_ridge_mci)
    MSE.append(MSEtemp)
    coefs.append(ridge_mci.coef_)
print (MSE)
    
ax = plt.gca()
ax.plot(alphas, MSE)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('Penalty Parameters (alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression MCI BRAAK56 Testing Set')
plt.show()  
####

ax = plt.gca()
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularisation (BRAAK C MCI)')
plt.axis('tight')
plt.show()
####

#applying Grid Search to find the best model and the best parameter
myparams = {'alpha': alphas}

#Nested Cross Validation for Ridge Regression
ncv_scores_ridge = []
outer_cv_ridge = KFold(n_splits=10)
X_mat_ridge=X_norm_train #check for NaN
y_ridge=y_norm_mci_train #
#error: cannot have no of splits greater than number of samples
for train_index, test_index in outer_cv_ridge.split(X_mat_ridge):
    tmp_train_ridge= X_mat_ridge.iloc[train_index,:]
    y_tmp_train_ridge = y_ridge.iloc[train_index]
    
    tmp_test_ridge = X_mat_ridge.iloc[test_index,:]
    y_tmp_test_ridge = y_ridge.iloc[test_index]
    inner_cv_ridge= KFold(n_splits=5)
    gs_ridge_mci = GridSearchCV(ridge_mci, myparams, scoring = 'neg_mean_squared_error',cv = 5)                        
    gs_ridge_mci = gs_ridge_mci.fit(X_norm_train, y_norm_mci_train)
    yhat_tmp_ridge = gs_ridge_mci.best_estimator_.predict(tmp_test_ridge)
    ncv_scores_ridge.append(mean_squared_error(yhat_tmp_ridge,y_tmp_test_ridge))
    
    #checking the best results after nested cross validation
    print(ncv_scores_ridge)
    best_accuracy = gs_ridge_mci.best_score_
    best_parameters_ridge = gs_ridge_mci.best_params_
    print(best_accuracy)
    print(best_parameters_ridge)
    print(gs_ridge_mci.cv_results_)
    
    #visualising the results
    
    ax = plt.gca()
    ax.plot(alphas, gs_ridge_mci.cv_results_['mean_test_score']/(-1)) 
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('Penalty Parameters (alphas)')
    plt.ylabel('MCI Mean Test Score')
    plt.title('Ridge Regression Nested Cross Validation MCI BRAAK 56')
    plt.show()  
    
#applying Ridge Regression for the best alpha parameter
ridgemci_best = Ridge(alpha = a, normalize = True)
ridgemci_best.fit(X_norm_train, y_norm_mci_train)
pred_ridgemci_best = ridgemci_best.predict(X_norm_test)
MSE_best = mean_squared_error(y_norm_mci_test, pred_ridgemci_best)
print(MSE_best)

##correlation coefficient for y test and ridge regression
corr_coef_ridge_mci,_ = pearsonr(y_tmp_test_ridge, yhat_tmp_ridge)
print('Pearsons correlation: %.3f' % corr_coef_ridge_mci)
#r=0.404


pyplot.scatter(y_tmp_test_ridge, yhat_tmp_ridge, color = 'red')
plt.xlabel('Test Values  (MCI BRAAK56)')
plt.ylabel('Predicted Values (MCI BRAAK56)')
plt.title('Ridge Regression Scatter Plot MCI BRAAK 56 Test Correlation Dataset')
pyplot.show()


#RBF Support Vector Machine for MCI BRAAK 12
#C = np.logspace(start = -5, stop = 0, num = 70 ) 
#this is a good value for MCI BRAAK56 but not for MCI 12
C = np.logspace(start = -3, stop = 5 , num = 40)
svr_mci_rbf = SVR(kernel = 'rbf', gamma = 'scale')
MSE = [] 
for a in C:
    svr_mci_rbf.set_params(C=a)
    svr_mci_rbf.fit(X_norm_train, y_norm_mci_train)
    pred_mci_rbf = svr_mci_rbf.predict(X_norm_test)
    MSEtemp = mean_squared_error(y_norm_mci_test, pred_mci_rbf)
    MSE.append(MSEtemp)

ax = plt.gca()
    
ax.plot(C, MSE)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('Regularisation Parameter C')
plt.ylabel('Mean Squared Error')
plt.title('RBF Support Vector Regression MCI Braak 56')
plt.show()  
#grid search for RBF SVR
gamma = np.logspace(-9,3,13)
#print(gamma)
myparams_rbf = {'C': C, 'gamma': gamma}

#RBF Nested Cross Validation
ncv_scores_rbf = []
outer_cv_rbf = KFold(n_splits=10)
X_mat_rbf=X_norm_train #check for NaN
y_rbf=y_norm_mci_train 
for train_index_rbf, test_index_rbf in outer_cv_rbf.split(X_mat_rbf):
    tmp_train_rbf = X_mat_rbf.iloc[train_index_rbf,:]
    y_tmp_train_rbf = y_rbf.iloc[train_index_rbf] #nan here
    
    tmp_test_rbf = X_mat_rbf.iloc[test_index_rbf,:]
    y_tmp_test_rbf = y_rbf.iloc[test_index_rbf]
    inner_cv_rbf = KFold(n_splits=5)
    gs_rbf_mci = GridSearchCV(svr_mci_rbf, myparams_rbf, scoring = 'neg_mean_squared_error',cv = inner_cv_rbf)
    gs_rbf_mci = gs_rbf_mci.fit(tmp_train_rbf, y_tmp_train_rbf)
    yhat_tmp_rbf = gs_rbf_mci.best_estimator_.predict(tmp_test_rbf)
    ncv_scores_rbf.append(mean_squared_error(yhat_tmp_rbf,y_tmp_test_rbf))
    best_accuracy = gs_rbf_mci.best_score_
    best_parameters_rbf_mci= gs_rbf_mci.best_params_
    print(best_accuracy)
    print(best_parameters_rbf_mci)
    print(gs_rbf_mci.cv_results_)

scores_rbf_mci = gs_rbf_mci.cv_results_['mean_test_score'].reshape(len(C),len(gamma))
ax = sns.heatmap(scores_rbf_mci)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
plt.xticks(np.arange(len(gamma)), gamma) 
ggg=np.round(np.log10(C),1)
plt.yticks(np.arange(len(C)), ggg)
plt.xlabel('gamma')
plt.ylabel('log10(C)')
plt.title('Performance RBF SVR for MCI BRAAK 56')
#side bar needs label as Mean Squared Error
plt.show()  

 
from scipy.stats import pearsonr
corr_coef_rbf_mci,_ = pearsonr(y_tmp_test_rbf, yhat_tmp_rbf)
print('Pearsons correlation: %.3f' % corr_coef_rbf_mci)
from matplotlib import pyplot
pyplot.scatter(y_tmp_test_rbf, yhat_tmp_rbf, color = 'green')
plt.xlabel('Test Values MCI(BRAAK 56)')
plt.ylabel('Predicted Values MCI (BRAAK 56)')
plt.title('RBF SVR Scatter Plot Test Correlation MCI Dataset')
pyplot.show()

##for context report:
gender_inclusivity_mci = results_bl_mcitest.PTGENDER.value_counts()
ethnic_inclusivity_mci = results_bl_mcitest.PTRACCAT.value_counts()
##

#Linear Support Vector Regression for MCI BRAAK12
from sklearn.svm import SVR
C = np.logspace(start = -5, stop = 0, num = 50 ) #make this a smaller range
#epsilon = [2,4]
print(C)
C
svr_lin = SVR(kernel = 'linear') #try rbf
MSE = [] 
for a in C:
    svr_lin.set_params(C=a)
    svr_lin.fit(X_norm_train, y_norm_mci_train)
    pred_lin_mci = svr_lin.predict(X_norm_test)
    MSEtemp = mean_squared_error(y_norm_mci_test, pred_lin_mci)
    MSE.append(MSEtemp)

ax = plt.gca()
    
ax.plot(C, MSE)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('C')
plt.ylabel('MSE')
plt.title('Linear Support Vector Regression MCI Braak 56')
plt.show()  

#applying Grid Search on linear SVR 
epsilon = [0.005,0.1, 0.15, 0.2]
myparams_SVR = {'C': C, 'epsilon': epsilon}

#Nested Cross Validation for Linear SVR
ncv_scores_lin = []
outer_cv_lin = KFold(n_splits=10)
X_mat_lin=X_norm_train #check for NaN
y_lin=y_norm_mci_train #
for train_index_lin, test_index_lin in outer_cv_lin.split(X_mat_lin):
    tmp_train_lin = X_mat_lin.iloc[train_index_lin,:]
    y_tmp_train_lin = y_lin.iloc[train_index_lin] #nan here
    
    tmp_test_lin = X_mat_lin.iloc[test_index_lin,:]
    y_tmp_test_lin = y_lin.iloc[test_index_lin]
    inner_cv_lin = KFold(n_splits=5)
    gs_lin_mci = GridSearchCV(svr_lin, myparams_SVR, scoring = 'neg_mean_squared_error',cv = 5)                      
    gs_lin_mci = gs_lin_mci.fit(X_norm_train, y_norm_mci_train)
    yhat_tmp_lin = gs_lin_mci.best_estimator_.predict(tmp_test_lin)
    ncv_scores_lin.append(mean_squared_error(yhat_tmp_lin,y_tmp_test_lin))
    
    best_accuracy = gs_lin_mci.best_score_
    best_parameters_lin_mci= gs_lin_mci.best_params_
    print(best_accuracy)
    print(best_parameters_lin_mci)
    print(gs_lin_mci.cv_results_)
    


scores_lin_mci = gs_lin_mci.cv_results_['mean_test_score'].reshape(len(C),len(epsilon))
ax = sns.heatmap(scores_lin_mci)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
plt.xticks(np.arange(len(epsilon)), epsilon) #need a smaller range so it looks ok
gg=np.round(np.log10(C),1)
plt.yticks(np.arange(len(C)), gg)
#plt.yticks(np.arange(len(C)), C)
plt.xlabel('epsilon')
plt.ylabel('log10(C)')
plt.title('Performance Linear SVR for MCI BRAAK 56')
#side bar needs label as Mean Squared Error
plt.show()  


#correlation coefficient for Linear SVR
corr_coef_lin,_ = pearsonr(y_tmp_test_lin, yhat_tmp_lin)
print('Pearsons correlation: %.3f' % corr_coef_lin)
#r=0.552
pyplot.scatter(y_tmp_test_lin, yhat_tmp_lin, color = 'blue')
plt.xlabel('Test Values  (BRAAK 56)')
plt.ylabel('Predicted Values (BRAAK56)')
plt.title('Linear SVR Scatter Plot Test Correlation')
pyplot.show()


#Random Forest Regression
#Fitting Regression to the Training Set
from sklearn.ensemble import RandomForestRegressor
n_estimators = [2**n for n in range(4,11)]
print (n_estimators)

max_features = [0.2, 0.4, 0.6, 0.8]
print(max_features)
rf_mciC= RandomForestRegressor(criterion = 'mse')
rf_mciC= RandomForestRegressor()

#Nested Cross Validation
myparams_rf = {'n_estimators': n_estimators, 'max_features': max_features}
print(myparams_rf)
ncv_scores_rfr = []
outer_cv_rfr = KFold(n_splits=10)
X_mat_rfr=X_norm_train #check for NaN
y_rfr=y_norm_mci_train
for train_index_rfr, test_index_rfr in outer_cv_rfr.split(X_mat_rfr):
    tmp_train_rfr = X_mat_rfr.iloc[train_index_rfr,:]
    y_tmp_train_rfr = y_rfr.iloc[train_index_rfr] #nan here
    
    tmp_test_rfr = X_mat_rfr.iloc[test_index_rfr,:]
    y_tmp_test_rfr = y_rfr.iloc[test_index_rfr]
    inner_cv_rfr = KFold(n_splits=5)
    grid_search_rf = GridSearchCV(rf_mciC,myparams_rf, scoring = 'neg_mean_squared_error',cv = 5 )
    grid_search_rf = grid_search_rf.fit(X_norm_train, y_norm_mci_train)
    yhat_tmp_rfr = grid_search_rf.best_estimator_.predict(tmp_test_rfr)
    ncv_scores_rfr.append(mean_squared_error(yhat_tmp_rfr,y_tmp_test_rfr))
    
scores_rf = grid_search_rf.cv_results_['mean_test_score'].reshape(len(n_estimators),len(max_features))


#seaborn heat plot for number of trees and the max features
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(scores_rf)
plt.xticks(np.arange(len(max_features)), max_features) #need a smaller range so it looks ok
plt.yticks(np.arange(len(n_estimators)), n_estimators)
plt.xlabel('Maximum Features')
plt.ylabel('Number of Trees')
plt.title('Performance Random Forest Regression (MCI Braak A)')
plt.show()


#grid then heat plot - max number of features - see the changes with the forest size 
gs_rf_mciC = GridSearchCV(rf_mciC, myparams_rf, scoring = 'neg_mean_squared_error',cv = 5)#change variable name                  
gs_rf_mciC = gs_rf_mciC.fit(X_norm_train, y_norm_mci_train)
best_parameters_rf_mciC= gs_rf_mciC.best_params_
print(best_parameters_rf_mciC)

pred_rf_mciC_best = gs_rf_mciC.best_estimator_.predict(X_norm_test)
MSE_best_mciC = mean_squared_error(y_norm_mci_test, pred_rf_mciC_best)
print(MSE_best_mciC)

scores_gs_rf = gs_rf_mciC.cv_results_['mean_test_score'].reshape(len(n_estimators),len(max_features))

#plotting the results for Grid Search Cross Validation
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(scores_gs_rf)
plt.xticks(np.arange(len(max_features)), max_features) #need a smaller range so it looks ok
plt.yticks(np.arange(len(n_estimators)), n_estimators)
plt.xlabel('Maximum Features')
plt.ylabel('Number of Trees')
plt.title('Performance Random Forest Regression (MCI Braak C)')
plt.show()

##correlation coefficient for y test and ridge regression
corr_coef_rf_mciC,_ = pearsonr(y_norm_mci_test, pred_rf_mciC_best)
print('Pearsons correlation: %.3f' % corr_coef_rf_mciC)
#r=0.340

s_corr_coef_rf_mciC,_=spearmanr(y_norm_mci_test, pred_rf_mciC_best)
print('Spearmans correlation: %.3f' %s_corr_coef_rf_mciC)

fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_mci_test, pred_rf_mciC_best, color = 'orange')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression Scatter Plot Test Set (MCI BRAAK C)')
pyplot.show()


