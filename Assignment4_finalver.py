#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 13:04:21 2021

@author: sayedakhan
"""

from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
from matplotlib import pyplot
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import spearmanr, pearsonr
import scipy.stats as stats
import pandas as pd
import numpy as np
from scipy.stats import shapiro
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, bartlett


adultincome = pd.read_csv('/Users/sayedakhan/Downloads/adult.csv')
adultincome_small = adultincome.sample(100)
list(adultincome_small)

####################### IDENTIFYING THE LEVELS FOR THE CHOSEN VARIABLES #######################

#1 factor AGE has 74 levels
adultincome.age.value_counts()
adultincome['age'].value_counts()
len(adultincome.age.value_counts())

#1 factor RACE has 5 levels
adultincome.race.value_counts()
adultincome['race'].value_counts()
len(adultincome.race.value_counts())

#1 factor OCCUPATION has 15 levels
adultincome.occupation.value_counts()
adultincome['occupation'].value_counts()
len(adultincome.occupation.value_counts())

# Data transformation on OCCUPATION column: replace '?' with 'Other'
adultincome = adultincome.replace('?', np.NaN)
adultincome['occupation'] = adultincome['occupation'].replace(np.NaN, 'Other')

#1 factor INCOME has 2 levels: <=50K and >50K 
adultincome.income.value_counts()
adultincome['income'].value_counts()
adultincome['income'] = adultincome['income'].replace('<=50K' , '0')
adultincome['income'] = adultincome['income'].replace('>50K' , '1')

adultincome['income'] = adultincome['income'].replace('0' , '<=50K')
adultincome['income'] = adultincome['income'].replace('1' , '>50K')


#1 factor WORKCLASS has 8 levels
adultincome.workclass.value_counts()
len(adultincome.workclass.value_counts())

#1 factor EDUCATION has 16 levels

adultincome.education.value_counts()
adultincome['educational-num'].value_counts()

len(adultincome.education.value_counts())

#1 factor HOURS PER WEEK has 96 levels
adultincome['hours-per-week'].value_counts()

level = adultincome[adultincome['education'] + [adultincome['educational-num']=='level']
                    
#COMBINE EDUCATION AND EDUCATIONAL-NUM 
adultincome['education-level'] = adultincome['educational-num'].astype(str) + '_' + adultincome['education']

################### QUESTION 1 #######################

# ONE CONTINUOUS DEPENDENT VARIABLE - INCOME

# THREE INDEPENDENT VARIABLES WITH >= 3 LEVELS - RACE, OCCUPATION, EDUCATION


################### QUESTION 2 ################### 
################### Conduct the proper pre (assumption) testing ################### 

# Is there a difference between the levels of race and education?
# DV ~ C(IV) + C(IV)


# Create a race chart

adultincomerace1 = adultincome[adultincome['race']=='White']
adultincomerace2 = adultincome[adultincome['race']=='Black']
adultincomerace3 = adultincome[adultincome['race']=='Asian-Pac-Islander']
adultincomerace4 = adultincome[adultincome['race']=='Amer-Indian-Eskimo']
adultincomerace5 = adultincome[adultincome['race']=='Other']


## Homogeneity of Variance 
## barlett test - RACE AND EDUCATION LEVEL
stats.bartlett(adultincomerace1['educational-num'],
               adultincomerace2['educational-num'],
               adultincomerace3['educational-num'],
               adultincomerace4['educational-num'],
               adultincomerace5['educational-num']
               )
                      )
#homogeneity not met- pvalue is less than 0


   
## barlett test - RACE AND HOURS-PER-WEEK
stats.bartlett(adultincomerace1['hours-per-week'],
               adultincomerace2['hours-per-week'],
               adultincomerace3['hours-per-week'],
               adultincomerace4['hours-per-week'],
               adultincomerace5['hours-per-week']
               )
#homogeneity not met- pvalue is less than 0

# check for kurtosis 

print(kurtosis(adultincomerace1['educational-num']))
print(kurtosis(adultincomerace2['educational-num']))
print(kurtosis(adultincomerace3['educational-num']))
print(kurtosis(adultincomerace4['educational-num']))
print(kurtosis(adultincomerace5['educational-num']))

## leptokurtic

# check for skewness(race)


print(skew((adultincomerace1['educational-num'])))
print(skew((adultincomerace2['educational-num'])))
print(skew((adultincomerace3['educational-num'])))
print(skew((adultincomerace4['educational-num'])))
print(skew((adultincomerace5['educational-num'])))

## negatively skewed

# histogram depicting income levels in white americans
plt.hist(adultincomerace1['income'])
plt.show()


# histogram depicting income levels in black americans
plt.hist(adultincomerace2['income'])
plt.show()



plt.hist(adultincome['occupation'])
plt.show()

adultincome['occupation'].value_counts()


               
################### QUESTION 3 ################### 
################### Conduct at least 3 1-way anova’s ################### 

# Is there an association between the number of hours worked per week and gender?


import statsmodels.stats.multicomp as mc
comp = mc.MultiComparison(adultincome['educational-num'], adultincome['race'])
post_hoc_res = comp.tukeyhsd()
tukey1way = pd.DataFrame(post_hoc_res.summary())

comp = mc.MultiComparison(adultincome['hours-per-week'], adultincome['gender'])
post_hoc_res = comp.tukeyhsd()
tukey1way = pd.DataFrame(post_hoc_res.summary())


# Is there an association between the number of hours worked per week and occupation?

comp = mc.MultiComparison(adultincome['hours-per-week'], adultincome['occupation'])
post_hoc_res = comp.tukeyhsd()
tukey1way = pd.DataFrame(post_hoc_res.summary())


# Is there an association between income and gender? **ERROR**
comp = mc.MultiComparison(adultincome['age'], adultincome['occupation'])
post_hoc_res = comp.tukeyhsd()
tukey1way = pd.DataFrame(post_hoc_res.summary())

