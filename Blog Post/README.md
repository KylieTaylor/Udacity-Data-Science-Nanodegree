# 3 Surprising Facts about Global Poverty and Life Expectancy Blog Post


### In this Udacity Data Scientist Nanodegree project, I write a blog post on "3 Surprising Facts about Global Poverty and Life Expectancy". 

### Find the blog post on Medium here:
https://kyliewtaylor.medium.com/3-surprising-facts-about-global-poverty-and-life-expectancy-39b2799ee058

## Libraries
In this project I used the following Python libraries.
``` 
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
sns.set(rc={'figure.figsize':(8,8)})

import matplotlib.pyplot as plt
%matplotlib inline

import plotly
import plotly.express as px

from sklearn.linear_model import LinearRegression, LassoCV
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
```

## Motivation for Project

I continually find myself gravitating to global poverty and socio-economic measures whenever posed with an open ended research project. Since I like to stay on brand, I decided to take a Data Scientists approach to interrogating the most current data on global poverty (GDP per capita) and life expectancy.
The analysis compares the 12 poorest countries to the 12 richest countries to effectively show the reader how dramatically different the metrics are for each group.
At this point it is almost tribal knowledge that global income disparity is massive - this analysis uses data to prove that the differences between rich and poor countries is far more than a simple income disparity story.

## Files 

All python code to run the analysis can be found in the Blog Analysis.ipynb file. Supporting files include the Data folder which houses all the raw data I used for the analyses. In addition, all visualizations generated in the analysis are kept in the Images folder. The project rubric is provided as reference.

## Summary

The 3 surprising facts are:
1) GDP per capita has remained stable for the poorest and richest countries from 2010 to 2015.
    Neither the poorest or the richest countires have saw increases in GDP per capita from 2010 to 2015.
    
2) Government expenditure on education has a positive relationship to life expectancy in the poorest countries.
    The correlation between government expenditure on primary education and life expectancy is 0.69. This is significantly higher than the same correlation for rich countries.
    
3) Global income disparity is as bad as people make it out to be… possibly worse.
    Majority of the world lives on much less than $25,000 USD per capita per year. 

## Acknowledgements

The World Bank for providing data. https://databank.worldbank.org/home.aspx

The SciPy library documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

The sklearn KNN Imputer documentation. https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html

The seaborn clustermap documentation. https://seaborn.pydata.org/generated/seaborn.clustermap.html














