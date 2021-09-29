# Udacity Data Science Nanodegree Capstone Project
## How to Create a ML Replica of an Engineer - A Case Study


### Kylie Taylor
### September, 2021

#### Prerequisites
```
from google.colab import drive
import pandas as pd
import numpy as np
import stat
import glob
import os
from tabulate import tabulate
from datetime import datetime
from sklearn.metrics import make_scorer, precision_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import sweetviz as sv
```

Access to Google Colab and Google Drive for data storage is also required for this project.

#### Motivation

A typical task for mechanical and process engineers is to identify when a machine is in a particular state. This project uses machine learning models trained on labelled data to predict times when machines have attained the desired state. Predictions made by the model are used to subset manufacturing data - a time intensive step engineers typical undertake when investigating data. The winning machine learning model is determined by the model that simultaneously achieves the highest precision, accuracy, and specificity. 

The data is sourced from manufacturing machines residing in a non-disclosed loaction and contains the values generated from five sensors attached to the machine ('A', 'B', 'C', 'D', and 'E' - renamed to anonymize), the timestamp from when each value was generated, and the binary outcome feature, 'label'. 
Data used for modeling was collected from Jan 6, 2021 to August 15, 2021. 
Data labelling was performed by industry experts. That portion of the project is not shared in this report, as it contains confidential information.


#### Repository Description

[Data](https://github.com/KylieTaylor/Udacity-Data-Science-Nanodegree/tree/main/Capstone/data) - folder containing all data used to train, validate, and test models

[Capstone.ipynb]() - Notebook containing the data processing, EDA, feature engineering, model training, and model comparisons and evaluation. Connects directly to a personal Google Drive to read in data, since it was an easy integration into Google Colab (platform used to train models).


#### Results

The "winning" model is the Gradient Boosting Classifier - as the out of sample predictions achieved the highest cummulative scores. With 80% precision, 78% accuracy, and 85% specificity, this model is the best at capturing true positives and true negatives compared to all other models. In context of this use case, detecting true positives is preffered over detecting false positives. 

![image](https://user-images.githubusercontent.com/47127996/135311654-21b0d84b-ae11-47e1-8f9c-72fb620b2fdc.png)


#### Acknowledgements

[Scikit Learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

[Scikit Learn Classification models](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

[Scikit Learn Pipelining](https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html)

[Scikit Learn Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)

[Sweetviz](https://www.kaggle.com/ahmettezcantekin/sweetviz-simple-and-quick-eda)



