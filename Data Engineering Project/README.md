# Disaster Response Pipeline Project

## Kylie Taylor
### May 2021

### Introduction

This project was completed for the Data Engineering course of the Udacity Data Scientist Nanodegree. The goal of the project was to create a Flask web application that allows users to interact with a machine learning model that classifies disaster response messages, with data provided by [Figure Eight](https://appen.com/).

### Data

All data for this project was provided by [Figure Eight](https://appen.com/) and can be found in the [Data](https://github.com/KylieTaylor/Udacity-Data-Science-Nanodegree/tree/main/Data%20Engineering%20Project/Data) folder. The raw data files include a .csv with message classification lables and another .csv with the corresponding messages. The data files were cleaned and merged to create a masterfile. This master file is saved as a SQL database object, called   `DisasterResponse.db`.

### Model

A model pipeline was used to train the final model to classifiy message categories. The pipeline vectorized the messsages using sklearn's [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), transformed the vectorized data using a [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html), then trained a [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) that classified messages into over 30 categories. A hyperparameter grid search using 3 fold cross validation was used to explore various random forest models and select the best performing. The best random forest model was trained using 1 minimum sample per leaf, 5 minimum splits, and 50 estimators, with no maximum depth. The precision, recall, f1-score and support are reported for the model.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
   
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    
    `cd app`
    
    `python run.py`
    
3. In another terminal type the following command.
    
    `env|grep WORK `
    
4. In a new web browser window, type in the following:
    
    `https://SPACEID-3001.SPACEDOMAIN`


### References
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://github.com/Blostrupsen/disaster_response_pipelines
https://github.com/xseibel/udacity_ds_p2
