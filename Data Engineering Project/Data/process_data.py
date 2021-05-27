import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def cat_to_binary(categories: pd.DataFrame) -> pd.DataFrame:
    '''
    A function that turns categorical data into binary
    ARGS: a data frame that contain category labels
    OUTPUT: a data frame that contain category labels as binary
    '''
    for column in categories:

        categories[column] = categories[column].map(
            lambda x: 1 if int(x.split("-")[1]) > 0 else 0 )
    return categories

def split_cat(categories: pd.DataFrame) -> pd.DataFrame:
    '''
    Creates dummy variables out of the labelled categories
    ARGS: a data frame that contain category labels
    OUTPUT: a pd.DataFrame with the Categories now as column headers
            and one-hot encoded data
    '''
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[[1]].values[0]
    categories.columns = [ x.split("-")[0] for x in row]
    categories = cat_to_binary(categories)
    return categories


def load_data(messages_filepath:str, categories_filepath:str) -> pd.DataFrame:
    '''
    This is a function that loads two datasets used in the disaster response
    models. Thw data sets are merged into one file.
    ARGS: messeges_filepath = the messages data frame
          categories_filepath = the data frame that contains labelled data
    OUTPUT: masterfile of messages and corresponding categories as a pd.DataFrame
    '''
    cat = split_cat(pd.read_csv(categories_filepath))
    messages = pd.read_csv(messages_filepath)
    return pd.concat([messages, cat], join="inner", axis=1)
    
  
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Cleans data and drops duplicates
    ARGS: a data frame that contains duplicates (or not)
    OUTPUT: a de-deuplicated pd.DataFrame
    '''
    return df.drop_duplicates()


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    '''
    Saves the database in an sql .db file
    ARGS: df: a pd.DataFrame 
          database_filename: the name of the path that the df is saved as
    OUTPUT: None, saves a .db object
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response_KT', engine, if_exists='replace', index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()