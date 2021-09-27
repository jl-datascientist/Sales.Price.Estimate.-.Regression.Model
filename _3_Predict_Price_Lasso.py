# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 18:10:46 2021

@author: J.Lecourt

Sales price estimation : Predict product price using the optimized model that has been saved
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
from joblib import load
import copy
import json
import xgboost as xgb

### Load the optimized model and the associated scaler coefficients ###
def load_model(model_selected):
    """
    Function to load the optimized model and the associated scaler coefficients (to rescale numerical data)
    Arguments:
        model_selected (str)
    Returns:
        model (sklearn.model)
        coeff (pd.DataFrame)
    """
    model_path = "models\\"+model_selected+".joblib"
    coeff_path = "models\\"+model_selected+"_rescale"+".csv"
    model = load(model_path)
    coeff = pd.read_csv(coeff_path)
    return model, coeff

### Convert the product data to the target structure used by the model (features) ###
def convert_data(df_init,num_coeff,num_col,cat_col):
    """
    Function to convert the categorical data and the numerical data to the final structure used by the model
    Arguments:
        df_init (pd.DataFrame)
        num_coeff (pd.DataFrame)
        num_col (list)
        cat_col (list)
    Returns:
        df_converted (pd.DataFram)
    """
#   Standardize numerical data using the same coeff used by the model => var[i] = (var[i] - var[i].mean()) / var[i].std()
    df_num = copy.deepcopy(df_init[num_col])
    for i in df_num.columns:
        df_num[i] = (df_num[i] - num_coeff[i].loc[0]) / num_coeff[i].loc[1]
#   Transform categorical data using the same columns used by the model => via get_dummies()
    df_cat = df_init[cat_col]
    df_converted = df_num
    for i in df_cat.columns:
        df_converted = df_converted.join(pd.get_dummies(df_cat[i], prefix=i),how = 'inner')
#   Get the initial structure of X_train / X_test (=> 698 columns) and fill the appropriate columns with the last converted product informations
    features_initial_struct = pd.read_json("products\\features_struct.json")
    for i in df_converted.columns:
        features_initial_struct[i] = df_converted[i]
    df_converted = features_initial_struct
    return df_converted

### Rescale the predicted prices ###
def rescale_target_prices(pred,num_coeff):
    """
    Function to rescale the target prices (using the same coeffs for numerical standardization)
    Arguments:
        pred (pd.DataFrame)
        num_coeff (pd.DataFrame)
    Returns:
        rescaled_pred (pd.DataFram)
    """
    moy_price = num_coeff['price'][0]
    ec_price =  num_coeff['price'][1]
    rescaled_pred = (pred * ec_price) + moy_price
    return rescaled_pred

### Evaluate the price predictions with the optimized model for a product list ###
if __name__ == '__main__':
#   Load the optimized model with associated coefficient (for rescaling data via the same standardisation)
    model_selected = "lasso_best_model_20210926"
    my_model, my_coeff = load_model(model_selected)
    print("\nModel selected :",model_selected)
#   Load the product list to analyse into a dataframe
    df_prod = pd.read_csv("products\\product_list_exple.csv")
    print("\nInitial product list :\n",df_prod)
    df_prod['price'] = pd.DataFrame(np.zeros(shape=(df_prod.shape[0],1)), columns=['price'])
#   Convert this product dataframe with the final categorical columns, and using the same standardisation for numerical columns (cf. data preparation for modeling)
    cat_col = ['category','sub_category','brand','material','color','quality']
    num_col = ['pop1','pop2']
    df_prod_converted = convert_data(df_prod,my_coeff,num_col,cat_col)
    print("\nConverted product list :\n",df_prod_converted)
#   Evaluate the prices with the selected model and rescale the target prices
    pred_prices = my_model.predict(df_prod_converted)
    pred_prices = rescale_target_prices(pred_prices,my_coeff)
    print("\nPrice prediction for each product of the list :\n")
    for i in list(range(pred_prices.size)):
        print("Product ",i,":",round(pred_prices[i],2))
        df_prod['price'][i] = round(pred_prices[i],2)
#   Return the predicted prices in a resulting CSV file with the initial product attributes
    df_prod.to_csv("products\\product_list_result.csv",index=False)
