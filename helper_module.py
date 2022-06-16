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

#Function to help visualize missing values
def help_missing(title = 'Features with missing values'):
    num_missing = train.isnull().sum().sort_values(ascending = False)
    pct = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    df_md = pd.concat([num_missing, pct], axis=1, keys=['Number', 'Percent'])
    
    #filtering for only the variables where there is 1 or more missing observations
    df_md.columns
    df_md = df_md.loc[df_md.Number > 0]
    has_missing = df_md['Number']
    
    #Making a chart
    fig, ax = plt.subplots()
    has_missing.plot.bar()
    p = ax.bar(x = has_missing.index, height = has_missing.values) 
    bottom, top = ax.get_ylim()
    ax.set_ylim(top = top*1.05)
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches(len(has_missing)/2, 4)
    plt.show()
    return df_md

#Function to help quickly look at a variable list of categorical features. For each feature, function gives number of missing values, pct of missing values, ANOVA effect size and p value, countplot and boxplot.
def look_cat(varlist):
    for var in varlist:
        data = pd.concat([train['log_SalePrice'], train[var]], axis=1)
        #info about missingness and value counts
        num_miss = data[var].isnull().sum()
        pct_miss = round(num_miss/data.shape[0]*100, 2)
        print(var + ' has ' + str(num_miss) + ' missing observations, equal to ' + str(pct_miss) + '%')
        print(train[var].value_counts())
        print('*'*50)
        #ANOVA
        temp_ = 'log_SalePrice ~ C(' + var + ')'
        model = ols(temp_, data = train).fit()
        anova_tbl = sm.stats.anova_lm(model, typ=2)
        effect_size = anova_tbl['sum_sq'][0]/(anova_tbl['sum_sq'][0]+anova_tbl['sum_sq'][1])
        p_val = round(anova_tbl['PR(>F)'][0], 5)
        print('ANOVA p-val and effect size for categories of ' + var + ' is', p_val, ' and ', round(effect_size, 5))
        print('*'*50)
        
         #Figures
        sns.set_theme(style="whitegrid", palette = 'husl')
        med = data.groupby([var])['log_SalePrice'].median().sort_values(ascending = False).reset_index()
        
        #seaborn Count Plot
        f, ax = plt.subplots(1, 2, figsize = (10,6))
        sns.countplot(x=data[var], data=data, order = list(med[var]), ax = ax[0])
        plt.xticks(rotation=45)
        #fig1.set(xlabel = var, ylabel = 'Count')
        sns.boxplot(x=var, y = 'log_SalePrice', order = list(med[var]), data=data, ax = ax[1])
        plt.xticks(rotation=45)
        #f.set(xlabel=var, ylabel='Log of Sale Price')
        
        yield f.show()      
        
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
def look_num(varlist):
    for var in varlist:
        
        #info about missingness
        num_miss = train[var].isnull().sum()
        pct_miss = round(num_miss/train.shape[0]*100, 2)
        print(train[var].describe())
        print('*'*50)
            
        #info about pearson's correlation
        corr = pearsonr(train[var], train['log_SalePrice'])
        print('Pearsons correlation (r):', round(corr[0],3))
        print('*'*50)
        
        #Figures
        sns.set_theme(style="whitegrid")

        #seaborn Histogram
        f, ax = plt.subplots(1, 2, figsize = (10,6))
        sns.histplot(x=var, data=train, ax = ax[0])
        #f.set(xlabel = var, ylabel = 'Count')
        plt.xticks(rotation = 45)
        
        #seaborn Scatterplot
        sns.scatterplot(x=var, y = 'log_SalePrice', data=train, ax = ax[1])
        plt.xticks(rotation=45)
        
        yield f.show()
'''        
#Function to help visualize missing values
def help_missing(title = 'Features with missing values'):
    num_missing = train.isnull().sum().sort_values(ascending = False)
    pct = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    df_md = pd.concat([num, pct], axis=1, keys=['Number', 'Percent'])
    
    #filtering for only the variables where there is 1 or more missing observations
    df_md.columns
    df_md = df_md.loc[df_md.Number > 0]
    has_missing = df_md['Number']
    
    #Making a chart
    fig, ax = plt.subplots()
    has_missing.plot.bar()
    p = ax.bar(x = has_missing.index, height = has_missing.values) 
    bottom, top = ax.get_ylim()
    ax.set_ylim(top = top*1.05)
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches(len(num)/2, 4)
    plt.show()
    return df_md'''

def get_dog():
    return 'Yes! Module loaded successfully! Sorry, no dogs!'