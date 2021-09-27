# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 11:40:33 2021

@author: J.Lecourt

Sales price estimation : Collect and prepare data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

### Collect data ###
def collect_data(dir_file):
    """
    Function to collect data from the CSV file
    Arguments:
        dir_file (str)
    Returns:
        df (pd.dataFrame)
    """
    df = pd.read_csv(dir_file)
    return df

### Clean and filter data ###
def clean_filter(df, col_to_drop, min_time, max_time):
    """
    Function to clean and filter data
    Arguments:
        df (pd.dataFrame)
        col_to_drop (list)
        min_time (int)
        max_time (int)
    Returns:
        df (pd.dataFrame)
    """
#   Pre-treatment : column to drop (inconsistent data)
    df = df.drop(col_to_drop, axis=1)
#   General treatment (drop inconsistent lines, filter on sold product, delete duplicates)
    df = df.dropna(how = 'any')
    df = df[df["sold"]==1]
    df = df.drop_duplicates()
#   Special end-treatment : filter on time_online to target prices that should be both attractive to sellers and buyers (business objective)
    df = df[(df["time_online"] >= min_time) & (df["time_online"] <= max_time)]
    return df

### Prepare data for modeling ###
def prepare_data(df, num_col, cat_col):
    """
    Function to normalise numerical data and to prepare categorical data for modeling
    Arguments:
        df (pd.dataFrame)
        num_col (list)
        cat_col (list)
    Returns:
        df (pd.dataFrame)
        scaler (preprocessing.StandardScaler)
    """
    df_num = df[num_col]
    df_cat = df[cat_col]
#   Standardize numerical data
    scaler = preprocessing.StandardScaler().fit(df_num)
    df_num = pd.DataFrame(scaler.transform(df_num), columns = df_num.columns, index = df_num.index)
#   Prepare categorical data and join with numerical data
    df_final = df_num
    for i in df_cat.columns:
        df_final = df_final.join(pd.get_dummies(df_cat[i], prefix=i),how = 'inner')
    return df_final, scaler

### Display dataframe treatment ###
if __name__ == '__main__':
#   Load data
    data_3 = collect_data("data\pricing_data.csv")
    print("Load data :\n",data_3)
#   Clean and filter data
    col_to_drop = ['model']
    data_3 = clean_filter(data_3, col_to_drop, 7, 30)
    print("\nAfter cleaning and filtering :",data_3.shape)
#   Prepare numerical and categorical data and display final dataset
    num_col = ['price','pop1','pop2']
    cat_col = ['category','sub_category','brand','material','color','quality']
    df_final, scaler = prepare_data(data_3, num_col, cat_col)
    print("\nAfter preparation for modeling :",df_final.shape,"\n")
    print(df_final)
#   Save the final dataset
    df_final.to_csv("data\\final_dataset.csv", index = False)
  


