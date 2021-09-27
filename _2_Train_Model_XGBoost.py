# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 13:27:47 2021

@author: J.Lecourt

Sales price estimation : Train and optimize model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _1_Collect_Data
from datetime import date
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from joblib import dump, load

### Load parameters for modeling ###
def load_parameters(dir_file):
    """
    Function to load parameters in order to train and optimize the model
    Arguments:
        dir_file (str)
    Returns:
        parameters (json)
    """
    with open(dir_file,"r") as f:
        parameters = json.load(f, encoding="utf-8")
    return parameters

### Get MAPE evaluation ###
def MAPE(pred_test,y_test):
    """
    Function to calculate the Mean Absolute Percentage Error (MAPE in %)
    Arguments:
        pred_test (pd.Series)
        y_test (pd.Series)
    Returns:
        mape (float)
    """
    error = []
    for i in list(range(pred_test.shape[0])):
        error.append(abs(y_test.iloc[i]-pred_test.iloc[i])/y_test.iloc[i])
    result = sum(error) / pred_test.shape[0] * 100
    return result

### Train, test and save the model ###
def train_test_model(df,random_state,parameters,scaler,num_col,dir_best_params,dir_save_model):
    """
    Function to train, optimize and save the model, and get the evaluation results
    Arguments:
        df (pd.dataFrame)
        random_state (int)
        parameters (json)
        scaler (preprocessing.StandardScaler)
        num_col (list)
        dir_best_params (str)
        dir_save_model (str)
    Returns:
        best_params (json)
        test_results (pd.DataFrame)
        rmse (float)
        mape (float)
    """
#   Split the dataset 80% 20% according to random_state to reproduce the case
    target = df['price']
    feats = df.drop(['price'], axis =1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=random_state)
#   Train the model with a gridsearch to get the best parameters
    model = xgb.XGBRegressor()
    grid_search = GridSearchCV(estimator = model, param_grid = parameters, cv = 3, n_jobs=-1, verbose = 1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
#   Evaluate the model with the best parameters and get the associated predictions
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(params=best_params,
                      dtrain=dtrain,
                      num_boost_round=999,
                      evals=[(dtrain,"Train"),(dtest, "Test")],
                      early_stopping_rounds=150
                      )
    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)
#   Archive each day the last best parameters found in a JSON file
    today = date.today().strftime("%Y%m%d")
    with open(dir_best_params+'_'+today+'.json', "w") as f_params:
        json.dump(best_params, f_params)
#   Evaluate the associated RMSE (performance indicator for regression model)
    rmse = []
    rmse.append(np.sqrt(mean_squared_error(pred_train, y_train)))
    rmse.append(np.sqrt(mean_squared_error(pred_test, y_test)))
#   Calculate the coefficient to rescale the data after standardisation
    rescale = pd.DataFrame(np.array([scaler.mean_, scaler.scale_]), columns = num_col)
    moy_price = rescale['price'][0]
    ec_price =  rescale['price'][1]
#   Evaluate the associated MAPE (performance percentage for regression model, relative indicator)
    test_results = pd.DataFrame({'expected_price': (y_test*ec_price)+moy_price,'predicted_price' : np.round((pred_test*ec_price)+moy_price)}, index = X_test.index)
    mape = MAPE(test_results["predicted_price"],test_results["expected_price"])
#   Save the optimized model and associated coefficient to rescale data
    rescale.to_csv(dir_save_model+'_'+today+'_rescale'+'.csv',index=False)
    dump(model,dir_save_model+'_'+today+'.joblib')
    return best_params, test_results, rmse, mape

### Display results after modeling and optimization ###
if __name__ == '__main__':
#   Collect and prepare data
    data_3 = _1_Collect_Data.collect_data("data\pricing_data.csv")
    data_3 = _1_Collect_Data.clean_filter(data_3, ['model'], 7, 30)
    num_col = ['price','pop1','pop2']
    cat_col = ['category','sub_category','brand','material','color','quality']
    df_final, scaler = _1_Collect_Data.prepare_data(data_3, num_col, cat_col)
    print("\nSelected dataframe size :",df_final.shape)
#   Load parameters to tune the model
    params = load_parameters("parameters\\xgboost_tuning_params.json")
    print("\nSelected parameters to tune the model :\n",params,"\n")
#   Train, optimize and save the model, and display results on test data
    best_params, test_results, rmse, mape = train_test_model(df_final,789,params,scaler,num_col,"parameters\\xgboost_best_params","models\\xgboost_best_model")
    print("\nBest parameters calculated after optimization :\n",best_params)
    print("\n -> RMSE on train data :",round(rmse[0],2))
    print("\n -> RMSE on test data :",round(rmse[1],2))
    print("\n -> Expected and predicted prices on test data :\n\n",test_results.reset_index(drop=True))
    print("\n -> MAPE on target prices (test data) :",round(mape,1),"%")
#   Export results on test data
    dir_result = "results\\xgboost_results_"+date.today().strftime("%Y%m%d")+".csv"
    test_results.to_csv(dir_result,index=False)
