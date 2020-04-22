# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 00:21:05 2020

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
X = results_bl.loc [:, results_bl.columns.str.endswith('VOLUME')] 
ICV_dummy = results_bl.loc[:, 'ICV_bl']
ICV_dummy.fillna(value=np.mean(ICV_dummy), inplace = True) #handling missing data
X_norm = X.apply(lambda x: x/ICV_dummy, axis=0)
#normalizing by the recommended reference region (white matter)
y_norm = results_bl.loc [:,'BRAAK12_SUVR']/ results_bl.loc[:, 'ERODED_SUBCORTICALWM_SUVR']  


#Splitting the data set intro Training Set and Testing Set
from sklearn.model_selection import train_test_split
X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X_norm, y_norm, test_size = 0.2)
#the training set contains a known output and the model learns on this data                  
#the test set tests our model's predictions based on the training set

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


#Box plots for All Diagnosis
#BRAAK56/INFERIOR
normalise_C = results.loc[:,'BRAAK56_SUVR'] / results.loc[:,'INFERIORCEREBELLUM_SUVR']
normalise_A = results.loc[:,'BRAAK12_SUVR']/results.loc[:,'INFERIORCEREBELLUM_SUVR']
normalise_B = results.loc[:,'BRAAK34_SUVR'] /results.loc[:,'INFERIORCEREBELLUM_SUVR']
normalise = pd.concat([normalise_A, normalise_B, normalise_C], axis = 1)
results_norm=pd.concat([normalise, results], axis = 1)
results_norm.rename(columns = {0:'BRAAK12_NORM'}, inplace =True)
results_norm.rename(columns = {1:'BRAAK34_NORM'}, inplace =True)
results_norm.rename(columns = {2:'BRAAK56_NORM'}, inplace =True)

#Boxplot for BRAAK 12
results_norm.boxplot(column = 'BRAAK12_NORM', by = 'DX'); #this works
#plt.xticks([1, 2, 3],['CN', 'MCI', 'Dementia'])
plt.xlabel('Diagnosis')
plt.ylabel('Braak A (ROI A) SUVR')
plt.show() 


#Boxplot for BRAAK 34
results_norm.boxplot(column = 'BRAAK34_NORM', by = 'DX'); #this works
plt.xlabel('Diagnosis')
plt.ylabel('Braak B (ROI B) SUVR')
plt.show() 


#Boxplot for BRAAK 56
results_norm.boxplot(column = 'BRAAK56_NORM', by = 'DX'); #this works
#plt.xticks([1, 2, 3],['CN', 'MCI', 'Dementia'])
plt.xlabel('Diagnosis')
plt.ylabel('Braak C (ROI C) SUVR')
plt.show() 



#Model 1: Ridge Regression
#Testing Ridge Regression for All Diagnosis
alphas =  np.logspace(start = -5, stop = 8, num = 80 ) #range assigned for 
print(alphas)
alphas
ridge_A= Ridge(normalize = True)
MSE = []
coefs = []
for a in alphas:
    ridge_A.set_params(alpha=a)
    ridge_A.fit(X_norm_train, y_norm_train)
    pred_ridge_A = ridge_A.predict(X_norm_test)
    MSEtemp = mean_squared_error(y_norm_test, pred_ridge_A)
    MSE.append(MSEtemp)
    coefs.append(ridge_A.coef_)
    
print (MSE)
print(coefs)

#plot of the MSE of penalty parameters before Cross-Validation    
ax = plt.gca()
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(alphas, MSE)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('Penalty Parameter (alpha)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Ridge Regression Testing Set (BRAAK A)')
plt.show()  

#plotting the Ridge coefficients as a function of the regularisation
ax = plt.gca()
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularisation')
plt.axis('tight')
plt.show()

#assigning the paramaters before Nested CV
myparams = {'alpha': alphas}
#Nested Cross Validation for Ridge Regression
#inner loop searches for best parameters with the lowest MSE
#outer loop is used for validation using best parameters

ncv_scores_ridge = []
outer_cv_ridge = KFold(n_splits=10)
X_mat_ridge_A=X_norm_train
y_ridge_A=y_norm_train 
for train_index, test_index in outer_cv_ridge.split(X_mat_ridge_A):
    tmp_train_ridge= X_mat_ridge_A.iloc[train_index,:]
    y_tmp_train_ridge = y_ridge_A.iloc[train_index]
    
    tmp_test_ridge = X_mat_ridge_A.iloc[test_index,:]
    y_tmp_test_ridge = y_ridge_A.iloc[test_index]
    inner_cv_ridge= KFold(n_splits=5)
    ngs_ridge_A= GridSearchCV(ridge_A, myparams, scoring = 'neg_mean_squared_error',cv = 5) #finds the best parameters                 
    ngs_ridge_A= ngs_ridge_A.fit(X_norm_train, y_norm_train)
    yhat_tmp_ridge_A = ngs_ridge_A.best_estimator_.predict(tmp_test_ridge) #makes a prediction for the current test data
    ncv_scores_ridge.append(mean_squared_error(yhat_tmp_ridge_A,y_tmp_test_ridge)) #computes performance
    
    #checking the best results after nested cross validation
    avg_MSE_ncv=sum(ncv_scores_ridge)/len(ncv_scores_ridge)
    print(avg_MSE_ncv)
    print(ngs_ridge_A.best_estimator_) #optimal penalty parameter
    print(ngs_ridge_A.best_score_) #MSE of the optimal alpha
    print(ngs_ridge_A.best_params_)
    print(ngs_ridge_A.cv_results_)
    
    #visualising the results
    
    ax = plt.gca()
    fig, ax = plt.subplots(figsize=(15,10))
    ax.plot(alphas, ngs_ridge_A.cv_results_['mean_test_score']/(-1)) 
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('Penalty Parameter (alphas)')
    plt.ylabel('Mean Test Score (MSE)')
    plt.title('Ridge Regression Nested CV Test Set (BRAAK A)')
    plt.show()  

    
gs_ridge_A = GridSearchCV(ridge_A, myparams, scoring = 'neg_mean_squared_error',cv = 10) #change variable name                  
gs_ridge_A = gs_ridge_A.fit(X_norm_train, y_norm_train)
pred_ridgeA_best = ngs_ridge_A.best_estimator_.predict(X_norm_test)
MSE_ridge=mean_squared_error(y_norm_test, pred_ridgeA_best)
print(MSE_ridge)
##correlation coefficient for y test and ridge regression
corr_coef_ridge_A,_ = pearsonr(y_norm_test, pred_ridgeA_best)
#corr_coef_ridge_A,_=pearsonr(y_tmp_test_ridge,yhat_tmp_ridge_A) #this calculated r on the last fold
print('Pearsons correlation: %.3f' % corr_coef_ridge_A)
#r=0.340

#visualising the results
#print(gs_ridge_mci.cv_results_)
best_est = pd.DataFrame(gs_ridge_A.best_estimator_.coef_).transpose()
#best_est.to_csv(r'C:\Users\Ruxandra Ion\Desktop\UCL LIFE\Year 4\Research Project\Braak A Baseline Ridge Test Set Best Estimator.csv')

##correlation coefficient for y test and ridge regression
corr_coef_ridge_A,_ = pearsonr(y_norm_test, pred_ridgeA_best)
print('Pearsons correlation: %.3f' % corr_coef_ridge_A)
#r=0.340

fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_test, pred_ridgeA_best, color = 'blue')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Ridge Regression Scatter Plot Test Set (BRAAK A)')
pyplot.show()



#Linear Support Vector Regression for BRAAK12
from sklearn.svm import SVR
C = np.logspace(start = -5, stop = 0, num = 50 ) #make this a smaller range
#epsilon = [2,4]
print(C)
C
svr_lin_A = SVR(kernel = 'linear') #try rbf
MSE = [] 
for a in C:
    svr_lin_A.set_params(C=a)
    svr_lin_A.fit(X_norm_train, y_norm_train)
    pred_lin_A = svr_lin_A.predict(X_norm_test)
    MSEtemp = mean_squared_error(y_norm_test, pred_lin_A)
    MSE.append(MSEtemp)

ax = plt.gca()
fig, ax = plt.subplots(figsize=(15,10)) 
ax.plot(C, MSE)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('Regularisation Parameter (C)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Linear Support Vector Regression Test Set (BRAAK A)')
plt.show()  

epsilon = [0.005,0.1, 0.15, 0.2]
myparams_SVR = {'C': C, 'epsilon': epsilon}
#Nested Cross Validation for Linear SVR
ncv_scores_lin = []
outer_cv_lin = KFold(n_splits=10)
X_mat_lin=X_norm_train #check for NaN
y_lin=y_norm_train #
for train_index_lin, test_index_lin in outer_cv_lin.split(X_mat_lin):
    tmp_train_lin = X_mat_lin.iloc[train_index_lin,:]
    y_tmp_train_lin = y_lin.iloc[train_index_lin] #nan here
    
    tmp_test_lin = X_mat_lin.iloc[test_index_lin,:]
    y_tmp_test_lin = y_lin.iloc[test_index_lin]
    inner_cv_lin = KFold(n_splits=5)
    ngs_lin_A = GridSearchCV(svr_lin_A, myparams_SVR, scoring = 'neg_mean_squared_error',cv = 5)                      
    ngs_lin_A = ngs_lin_A.fit(X_norm_train, y_norm_train)
    yhat_tmp_lin = ngs_lin_A.best_estimator_.predict(tmp_test_lin)
    ncv_scores_lin.append(mean_squared_error(yhat_tmp_lin,y_tmp_test_lin))
    
    best_accuracy = ngs_lin_A.best_score_
    best_parameters_lin_A= ngs_lin_A.best_params_
    print(best_accuracy)
    print(best_parameters_lin_A)
    print(ngs_lin_A.cv_results_)
    
    
scores_lin_A = ngs_lin_A.cv_results_['mean_test_score'].reshape(len(C),len(epsilon))
fig, ax = plt.subplots(figsize=(15,10)) 
ax = sns.heatmap(scores_lin_A)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
plt.xticks(np.arange(len(epsilon)), epsilon) 
gg=np.round(np.log10(C),1)
print(gg)
plt.yticks(np.arange(len(C)), gg)
plt.xlabel('epsilon')
plt.ylabel('log10(C)')
plt.title('Performance Linear SVR (BRAAK A) ')
#side bar needs label as Mean Squared Error
plt.show()  


gs_lin= GridSearchCV(svr_lin_A, myparams_SVR, scoring = 'neg_mean_squared_error',cv = 5)                      
gs_lin= gs_lin.fit(X_norm_train, y_norm_train)
best_parameters_linA=gs_lin.best_params_

pred_lin_A_best = ngs_lin_A.best_estimator_.predict(X_norm_test)
MSE_best = mean_squared_error(y_norm_test, pred_lin_A_best)
print(MSE_best)

#Saving results for brainpainter
best_est_lin_A = pd.DataFrame(gs_lin.best_estimator_.coef_)
best_est.to_csv(r'C:\Users\Ruxandra Ion\Desktop\UCL LIFE\Year 4\Research Project\Braak A Baseline Linear SVR Test Set Best Estimator.csv')

##correlation coefficient for y test and Linear SVR


corr_coef_lin_A,_ = pearsonr(y_norm_test, pred_lin_A_best)
print('Pearsons correlation: %.3f' % corr_coef_lin_A)
#r=0.316

fig, ax = plt.subplots(figsize=(15,10)) 
pyplot.scatter(y_norm_test, pred_lin_A_best, color = 'red')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Linear SVR Scatter Plot Test Set (BRAAK A)')
pyplot.show()

#RBF SVR
C = np.logspace(start = -3, stop = 5 , num = 40)
print(C)
svr_rbf_A = SVR(kernel = 'rbf', gamma = 'scale')
MSE = [] 
for a in C:
    svr_rbf_A.set_params(C=a)
    svr_rbf_A.fit(X_norm_train, y_norm_train)
    pred_rbf_A = svr_rbf_A.predict(X_norm_test)
    MSEtemp = mean_squared_error(y_norm_test, pred_rbf_A)
    MSE.append(MSEtemp)
    
fig, ax = plt.subplots(figsize=(15,10))   
ax = plt.gca()
ax.plot(C, MSE)
ax.set_xscale('log')
plt.axis('tight')
ax.set_xscale('log')
plt.xlabel('Regularisation Parameter (C)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('RBF Support Vector Regression Test Set (BRAAK A)')
plt.show()  

#Nested Cross Validation
gamma = np.logspace(-9,3,13)
print(gamma)
myparams_rbf = {'C': C, 'gamma': gamma}
ncv_scores_rbf = []
outer_cv_rbf = KFold(n_splits=10)
X_mat_rbf=X_norm_train #check for NaN
y_rbf=y_norm_train #
#error: cannot have no of splits greater than number of samples
for train_index_rbf, test_index_rbf in outer_cv_rbf.split(X_mat_rbf):
    tmp_train_rbf = X_mat_rbf.iloc[train_index_rbf,:]
    y_tmp_train_rbf = y_rbf.iloc[train_index_rbf] #nan here
    
    tmp_test_rbf = X_mat_rbf.iloc[test_index_rbf,:]
    y_tmp_test_rbf = y_rbf.iloc[test_index_rbf]
    inner_cv_rbf = KFold(n_splits=5)
    ngs_rbf_A = GridSearchCV(svr_rbf_A, myparams_rbf, scoring = 'neg_mean_squared_error',cv = inner_cv_rbf)
    ngs_rbf_A = ngs_rbf_A.fit(tmp_train_rbf, y_tmp_train_rbf)
    yhat_tmp_rbf = ngs_rbf_A.best_estimator_.predict(tmp_test_rbf)
    ncv_scores_rbf.append(mean_squared_error(yhat_tmp_rbf,y_tmp_test_rbf))

scores_rbf_mci = ngs_rbf_A.cv_results_['mean_test_score'].reshape(len(C),len(gamma))
fig, ax = plt.subplots(figsize=(30,20)) 
ax = sns.heatmap(scores_rbf_mci)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
plt.xticks(np.arange(len(gamma)), gamma) 
ggg=np.round(np.log10(C),1)
plt.yticks(np.arange(len(C)), ggg)
plt.xlabel('gamma')
plt.ylabel('log10(C)')
plt.title('Performance RBF Support Vector Regression (BRAAK A)')
#side bar needs label as Mean Squared Error
plt.show()  


gs_rbf_A = GridSearchCV(svr_rbf_A, myparams_rbf, scoring = 'neg_mean_squared_error',cv = 5)                      
gs_rbf_A = gs_rbf_A.fit(X_norm_train, y_norm_train)
pred_rbf_A_best = gs_rbf_A.best_estimator_.predict(X_norm_test)
best_parameters_rbf= ngs_rbf_A.best_params_
print(best_parameters_rbf)
MSE_best_rbf = mean_squared_error(y_norm_test, pred_rbf_A_best)
print(MSE_best_rbf)


from scipy.stats import pearsonr
corr_coef_rbf_A,_ = pearsonr(y_norm_test, pred_rbf_A_best)
print('Pearsons correlation: %.3f' % corr_coef_rbf_A)

from matplotlib import pyplot

fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_test, pred_rbf_A, color = 'green')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('RBF Support Vector Regression Scatter Plot Test Set (BRAAK A)')
pyplot.show()

###
#Random Forest Regression
#Fitting Regression to the Training Set
from sklearn.ensemble import RandomForestRegressor
n_estimators = [2**n for n in range(4,11)]
print (n_estimators)

max_features = [0.2, 0.4, 0.6, 0.8]
print(max_features)
rf = RandomForestRegressor(criterion = 'mse')
rf= RandomForestRegressor()

myparams_rf = {'n_estimators': n_estimators, 'max_features': max_features}
print(myparams_rf)
ncv_scores_rfr = []
outer_cv_rfr = KFold(n_splits=10)
X_mat_rfr=X_norm_train #check for NaN
y_rfr=y_norm_train
for train_index_rfr, test_index_rfr in outer_cv_rfr.split(X_mat_rfr):
    tmp_train_rfr = X_mat_rfr.iloc[train_index_rfr,:]
    y_tmp_train_rfr = y_rfr.iloc[train_index_rfr] #nan here
    
    tmp_test_rfr = X_mat_rfr.iloc[test_index_rfr,:]
    y_tmp_test_rfr = y_rfr.iloc[test_index_rfr]
    inner_cv_rfr = KFold(n_splits=5)
    grid_search_rf = GridSearchCV(rf,myparams_rf, scoring = 'neg_mean_squared_error',cv = 5 )
    grid_search_rf = grid_search_rf.fit(X_norm_train, y_norm_train)
    yhat_tmp_rfr = grid_search_rf.best_estimator_.predict(tmp_test_rfr)
    ncv_scores_rfr.append(mean_squared_error(yhat_tmp_rfr,y_tmp_test_rfr))
    
scores_rf = grid_search_rf.cv_results_['mean_test_score'].reshape(len(n_estimators),len(max_features))

avg_MSE_ncv_rfr=sum(ncv_scores_rfr)/len(ncv_scores_rfr)

#seaborn heat plot for number of trees and the max features
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(scores_rf)
plt.xticks(np.arange(len(max_features)), max_features) #need a smaller range so it looks ok
plt.yticks(np.arange(len(n_estimators)), n_estimators)
plt.xlabel('Maximum Features')
plt.ylabel('Number of Trees')
plt.title('Performance Random Forest Regression')
plt.show()


#grid then heat plot - max number of features - see the changes with the forest size 
gs_rf_A = GridSearchCV(rf, myparams_rf, scoring = 'neg_mean_squared_error',cv = 5)                 
gs_rf_A = gs_rf_A.fit(X_norm_train, y_norm_train)
best_parameters_rf_A= grid_search_rf.best_params_
print(best_parameters_rf_A)

pred_rfA_best = grid_search_rf.best_estimator_.predict(X_norm_test)
MSE_best = mean_squared_error(y_norm_test, pred_rfA_best)
print(MSE_best)

#visualising the results

##correlation coefficient for y test and ridge regression
corr_coef_rf_A,_ = pearsonr(y_norm_test, pred_rfA_best)
print('Pearsons correlation: %.3f' % corr_coef_rf_A)

s_corr_coef_rf_A,_=spearmanr(y_norm_test, pred_rfA_best)
print('Spearmans correlation: %.3f' %s_corr_coef_rf_A)

fig, ax = plt.subplots(figsize=(15,10))
pyplot.scatter(y_norm_test, pred_rfA_best, color = 'orange')
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression Scatter Plot Test Set (BRAAK A)')
pyplot.show()

