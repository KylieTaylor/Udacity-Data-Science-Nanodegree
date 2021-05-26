# Disaster Response Pipeline Project

## Kylie Taylor
### May 2021

### Introduction

This project was completed for the Data Engineering course of the Udacity Data Scientist Nanodegree. The goal of the project was to create a Flask web application that allows users to interact with a machine learning model that classifies disaster response messages, with data provided by [Figure Eight](https://appen.com/).

### Data

All data for this project was provided by [Figure Eight](https://appen.com/) and can be found in the [Data](https://github.com/KylieTaylor/Udacity-Data-Science-Nanodegree/tree/main/Data%20Engineering%20Project/Data) folder. The raw data files include a .csv with message classification lables and another .csv with the corresponding messages. The data files were cleaned and merged to create a masterfile. This master file is saved as a SQL database object, called   `DisasterResponse.db`.

### Model


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3. In another terminal type the following command.
    `env|grep WORK `
    
4. In a new web browser window, type in the following:
    `https://SPACEID-3001.SPACEDOMAIN`



