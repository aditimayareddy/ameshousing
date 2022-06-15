#Helper module with functions for Ames Regression Project
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
#Functions needed

train = pd.read_csv('./data/raw/train.csv')
train['log_SalePrice'] = np.log(train['SalePrice'])

#Function to help quickly look at a variable list of categorical features. For each feature, function gives number of missing values, pct of missing values, countplot and boxplot.
def look_cat(varlist):
    for var in varlist:
        data = pd.concat([train['log_SalePrice'], train[var]], axis=1)
        #info about missingness and value counts
        num_miss = data[var].isnull().sum()
        pct_miss = round(num_miss/data.shape[0]*100, 2)
        print(var + ' has ' + str(num_miss) + ' missing observations, equal to ' + str(pct_miss) + '%')
        print(train[var].value_counts())
        
         #Figures
        sns.set_theme(style="whitegrid")
        med = data.groupby([var])['log_SalePrice'].median().sort_values(ascending = False).reset_index()
        
         #seaborn Count Plot
        f, ax = plt.subplots(figsize = (6,4))
        fig1 = sns.countplot(x=data[var],
                      data=data,
                     order = list(med[var]));
        fig1.set(xlabel = var, ylabel = 'Count')
        
        
        f, ax = plt.subplots(figsize = (6,4))
        fig = sns.boxplot(x=var,
                          y = 'log_SalePrice',
                          order = list(med[var]),
                          data=data);
        plt.xticks(rotation=45)
        fig.set(xlabel=var, ylabel='Log of Sale Price')
        
        yield fig
        
        
#Function to help quickly look at a variable list of numerical features. For each feature, function gives number of missing values, pct of missing values, histogram and scatterplot.
def look_num(varlist):
    for var in varlist:
        
        #info about missingness
        num_miss = train[var].isnull().sum()
        pct_miss = round(num_miss/train.shape[0]*100, 2)
        print(var + ' has ' + str(num_miss) + ' missing observations, equal to ' + str(pct_miss) + '%')
        print(train[var].describe())
        
        #info about pearson's correlation
        #corr = pearsonr(train[var], train['log_SalePrice'])
        #print('Pearsons correlation (r):', round(corr[0],3))
        
        #Figures
        sns.set_theme(style="whitegrid")

        #seaborn Histogram
        f, ax = plt.subplots(figsize = (6,4))
        fig1 = sns.histplot(x=var,
                      data=train);
        fig1.set(xlabel = var, ylabel = 'Count')
        
        #seaborn Scatterplot
        f, ax = plt.subplots(figsize = (6,4))
        fig = sns.scatterplot(x=var,
                          y = 'log_SalePrice',
                          data=train);
        plt.xticks(rotation=45)
        fig.set(xlabel=var, ylabel='Log of Sale Price')
        
        yield fig
        
#Function for second pass in-depth look at numerical features
def look_num2(varlist):
    for var in varlist:
        
        #info about missingness
        num_miss = train[var].isnull().sum()
        pct_miss = round(num_miss/train.shape[0]*100, 2)
        print(train[var].describe())
        
        #info about pearson's correlation
        corr = pearsonr(train[var], train['log_SalePrice'])
        print('Pearsons correlation (r):', round(corr[0],3))
        
        #Figures
        sns.set_theme(style="whitegrid")

        #seaborn Histogram
        f, ax = plt.subplots(figsize = (6,4))
        fig1 = sns.histplot(x=var,
                      data=train);
        fig1.set(xlabel = var, ylabel = 'Count')
        
        #seaborn Scatterplot
        f, ax = plt.subplots(figsize = (6,4))
        fig = sns.scatterplot(x=var,
                          y = 'log_SalePrice',
                          data=train);
        plt.xticks(rotation=45)
        fig.set(xlabel=var, ylabel='Log of Sale Price')
        
        yield fig