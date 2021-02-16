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

I continually find myself gravitating to global poverty and other socio-economic measures whenever posed with an open ended research project. As I like to stay on brand, I decided to take a Data Scientists approach to interrogating the most current data on global poverty (GDP per capita) and life expectancy.
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

## CRISP DM Outline

### Research Understanding

Gain a brief understanding of what most recently global socioeconomic data reveals about school expenditure and enrollment, GDP per capita, and life expectancy. Three key topics are explored: how GDP per capita is changinf from year to year in poor countries in comparison to rich countries, the relationship that edcuation expenditure and school enrollment has on GDP per capita and life expectancy, and how to countries cluster together for GDP per capita and life expectancy.

### Data Understanding

All data for this project was sourced from the World Bank. Their data repository can be found here: https://data.worldbank.org/

The data set used for the analyses contains 217 countries, and the following features.

    - 'SP.DYN.LE00.IN' : 'Life expectancy at birth, total (years)',
    - 'SE.ENR.PRSC.FM.ZS' : 'School enrollment, primary and secondary (gross), gender parity index (GPI)',
    - 'NY.GDP.MKTP.CD' : 'GDP (current US$)',
    - 'NY.GDP.PCAP.CD' : 'GDP per capita (current US$)',
    - 'SE.XPD.TOTL.GD.ZS' : 'Government expenditure on education as percent of GDP',
    - 'SE.ENR.PRIM.FM.ZS' : 'Gross enrollment ratio, primary, gender parity index (GPI)'

Each feature is reported for the years 2010 to 2015. The GDP per capita is used as a proxy for poverty.

### Prepare Data

Data was prepared by removing features that were not used in the analysis and removing columns that contained no information. The data set was inspected for missing values, which is quite commmon for data coming from an open data source such as World Bank data. Not many additional feature transformations or feautre engineering was required for this project.

Data was split into subsets of the 12 poorest countries and 12 richest countries. This was done to stream line the analysis, as well as to demonstrate how large gloabl discrepancies can be. 

### Data Modeling

#### Fact 1

This section provides a simple visual inspection of GDP per Capita from 2010 to 2015. The rate of poverty in all countries has been stable over the years 2010 to 2015 for both the 12 poorest and 12 richest countries. The GDP per capita is reported in USD in adjusted to the 2021 price level.

#### Fact 2

This section calculates the correlation between a countries' education expenditure and school enrollment rates, and GDP per capita and life expectancy. The lag of school expenditure and enrollment are used in the correlations, becuase the effects of spending and enrollment in 2015 will not be seen in 2015. One MAJOR caveat is that we may not be able to see a reliable relationship with a 5 year lag.

#### Fact 3

This section clusters countries by GDP per capita and life expectancy for the years 2010 to 2015. In order to successfully run the clustering algorithm, missing data needs to be dealt with. Missing values were imputed using sklearn's KNN Imputer. The imputer uses 2 nearest neighbors (in this case, neighbors are countries) to calculate the mean value of a either GDP per capita or life expectancy. Two nearest neighbors were chosen to best resemble the uniqueness of individual countries. Not all countries should be modeled the same, therefore using 2 nearest neighbors encourages the algorithm to keep some variability within countries. If I were to hypothetically use 10 nearest neighbors, I would be making the assumption that countries are more similar in GDP per capita and life expectancy, than using 2 nearest neighbors.

A centroid clustering algorithm is used to cluster the various countries into their respective groups. Results of the clustering was displayed through a heatmap.

### Evaluation

This analysis barely brushed the surface of the global poverty and life expectancy landscape, but it does leverage real world data to show that the gloabl social conditions are radically different between the wealthiest and most impoverished countries.  

## Acknowledgements

The World Bank for providing data. https://databank.worldbank.org/home.aspx

The SciPy library documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

The sklearn KNN Imputer documentation. https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html

The seaborn clustermap documentation. https://seaborn.pydata.org/generated/seaborn.clustermap.html














