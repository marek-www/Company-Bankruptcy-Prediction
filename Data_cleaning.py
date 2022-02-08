# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:42:30 2021

@author: Marek
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats 
import pandas as pd
import seaborn as sns
# import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

filename = 'data.csv'
raw_data = pd.read_csv(filename)
data = raw_data.copy()

#==== Check categorical values
data.info()
cat_val = data.dtypes

#==== Check missing values
missing_val = data.isnull().sum() #no missing values

#==== Check duplicated rows
row_duplicates = data.duplicated().sum() #no duplicated rows

#==== Check duplicated values in columns
#trans_data = data.T
data = data.T.drop_duplicates().T
# almost_duplicates = data.iloc[:,:100].duplicated().sum()
data = data.drop([2605, 5015], axis = 0).reset_index(drop = True)

#==== Zero vs Non-zero values
non_zero_val = data['Bankrupt?'].value_counts() #High difference between positive and negative cases

#==== Statistical thresholds for each category
mean = data.groupby('Bankrupt?').mean()
median = data.groupby('Bankrupt?').median()
diff_mean = pd.DataFrame()
diff_median = pd.DataFrame()
diff = pd.DataFrame()
diff_mean['Mean difference'] = mean.iloc[0,:]/mean.iloc[1,:]*100-100
diff_median['Median difference'] = median.iloc[0,:]/median.iloc[1,:]*100-100
# diff_mean = diff_mean.sort_values(by = ['Mean difference'], ascending = False)
# diff_median = diff_median.sort_values(by = ['Median difference'], ascending = False)
diff['Mean difference'] = diff_mean['Mean difference']
diff['Median difference'] = diff_median['Median difference']
diff = diff.sort_values(by = ['Mean difference'], ascending = False)

small_diff = diff[(diff['Mean difference'] < 3) & (diff['Mean difference'] > -3) & 
                  (diff['Median difference'] < 3) & (diff['Median difference'] > -3)].T
#Small difference of mean/median values between positive and negative cases

# #==== Plots
plot_data = data.drop(columns = small_diff.columns)
col_names = list(plot_data.columns)
col_names2 = list(data.columns)
# plt.figure(dpi=300)
# ax = plt.axes()
# ax.set(xlabel='Net Value Per Share (A)',
#         ylabel='Net Value Per Share (B) / (C)',
#         title='Net Value Per Share depenadance');
# ax.scatter(data[' Net Value Per Share (A)'], data[' Net Value Per Share (B)'])
# ax.scatter(data[' Net Value Per Share (A)'], data[' Net Value Per Share (C)'])
# ax.legend([' Net Value Per Share (B)', ' Net Value Per Share (C)'])

data = data.drop(columns = [' Net Value Per Share (B)', ' Net Value Per Share (C)'])

# sns.set_context('talk')
# sns.pairplot(plot_data, hue='Bankrupt?')

ROA_check = data[[' ROA(A) before interest and % after tax', ' ROA(B) before interest and depreciation after tax',
                  ' ROA(C) before interest and depreciation before interest']]
# plt.figure(dpi=300)
# sns.set_context('talk', font_scale=0.35)
# sns.pairplot(ROA_check)

data = data.drop(columns = [' ROA(B) before interest and depreciation after tax',
                            ' ROA(C) before interest and depreciation before interest'])

Depend_check = data.iloc[:,22:24]
# sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
# sns.set_context('talk', font_scale=0.35)
# sns.pairplot(Depend_check)

data = data.drop(columns = [' Regular Net Profit Growth Rate'])

# #==== Skew
skew_limit = 0.75 # define a limit above which we will log transform
skew_vals = data.skew().sort_values(ascending = False).to_frame().rename(columns={0:'Skew'}).query('abs(Skew) > {}'.format(skew_limit))
for col in skew_vals.index.values:
    if col == 'Bankrupt?':
        continue
    data[col] = data[col].apply(np.log1p)

plot_data = data.drop(columns = small_diff.columns, errors = 'ignore')
# sns.set_context('talk')
# sns.pairplot(plot_data, hue='Bankrupt?')

# #==== Hypothesis assignment
hip_data = data[['Bankrupt?', ' Cash Flow to Sales', ' Operating Profit Rate', ' Working capitcal Turnover Rate']]
hip_stats = pd.DataFrame([hip_data.mean(), hip_data.std(), hip_data.var()], 
                          index=['Mean', 'Std. dev', 'Variance']).drop(columns = ['Bankrupt?'])
# =============================================================================
# # sns.set_context('talk', font_scale=0.35)
# # sns.pairplot(hip_data, hue='Bankrupt?')
# # plt.figure(dpi=300)
# # x = np.linspace(0.48, 0.55, 1000)
# # plt.plot(x, stats.norm.pdf(x, hip_stats.iloc[0,0], hip_stats.iloc[1,0]))
# # plt.title('Cash Flow to Sales')
# 
# # plt.figure(dpi=300)
# # x = np.linspace(0.66, 0.725, 1000)
# # plt.plot(x, stats.norm.pdf(x, hip_stats.iloc[0,1], hip_stats.iloc[1,1]), 'y')
# # plt.title('Operating Profit Rate')
# 
# # plt.figure(dpi=300)
# # x = np.linspace(0.44, 0.493, 1000)
# # plt.plot(x, stats.norm.pdf(x, hip_stats.iloc[0,2], hip_stats.iloc[1,2]), 'r')
# # plt.title('Working capitcal Turnover Rate')
# 
# # plt.figure(dpi=300)
# # x = hip_data.iloc[:,3].sort_values()
# # plt.plot(x, stats.norm.pdf(x, hip_stats.iloc[0,2], hip_stats.iloc[1,2]), 'r')
# # plt.title('Working capitcal Turnover Rate')
# # plt.xlim(0.44, 0.5)
# =============================================================================

# plt.figure(dpi=150)
# x = hip_data.iloc[:,1]
# plt.hist(x, bins=np.arange(min(x), max(x) + 0.000001, 0.000001), density = True)
# plt.title('Cash Flow to Sales')
# plt.xlim(0.51365, 0.5139)

# plt.figure(dpi=150)
# x = hip_data.iloc[:,2]
# plt.hist(x, bins=np.arange(min(x), max(x) + 0.00001, 0.00001), density = True)
# plt.title('Operating Profit Rate')
# plt.xlim(0.6922, 0.693)

# plt.figure(dpi=300)
# x = hip_data.iloc[:,3]
# plt.hist(x, bins=np.arange(min(x), max(x) + 0.00001, 0.00001), density = True)
# plt.title('Working capitcal Turnover Rate')
# plt.xlim(0.466, 0.4666)

#==== Cash flow to sales hypothesis testing
# plt.figure(dpi=300)
# plt.xlim(0.51365, 0.5139)
# sns.ecdfplot(x = hip_data.iloc[:,1])

# ecdf = ECDF(hip_data.iloc[:,1])
# p_5 = ecdf(0.51373348)
# p_95 = ecdf(0.51380497)

#==== Operating Profit Rate hypothesis testing
plt.figure(dpi=300)
plt.xlim(0.6922, 0.693)
sns.ecdfplot(x = hip_data.iloc[:,2])
sns.set_style("whitegrid")

ecdf = ECDF(hip_data.iloc[:,2])
p_5 = ecdf(0.6925075)
p_95 = ecdf(0.69277855)

test = hip_data.mean()
x = hip_data.iloc[:,2].sort_values()

hip_mean = data[['Bankrupt?',' Operating Profit Rate']].groupby('Bankrupt?').mean()

p_mean = ecdf(hip_mean.iloc[1,0])
