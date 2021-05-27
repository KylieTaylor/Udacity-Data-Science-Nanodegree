import sys
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, scorer, f1_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, HashingVectorizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
from typing import Tuple, List
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    '''
    Loads data into a pd.DataFrame
    ARGS: database_filepath: the path of the database, not including .db
    OUTPUT: X = the input features, this is 'messages'
            y = categories dummy variables
            categories = a list of categories
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_response_KT",engine) 
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'],  axis=1).astype(float)
    categories = y.columns.values
    return X, y, categories                  

                           
def tokenize(text) -> List[str]:
    '''
    Tokenizing function
    ARGS: Text string
    OUTPUT: List of tokens
    '''                       
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(t).lower().strip() for t in tokens]
                           

def build_model() -> GridSearchCV:
    '''
    Contains a model pipeline and hyperparameter grid search
    ARGS: None
    OUTPUT: model with the best hyperparameters
    '''
    pipe = Pipeline([('cvet', CountVectorizer(tokenizer=tokenize)),
                    ('tfdif', TfidfTransformer()),
                    ('rfc', RandomForestClassifier())
                    ])
    parameters = {'rfc__max_depth':[5, None],
              'rfc__n_estimators':[15, 50],
              'rfc__min_samples_split':[5, 10],
              'rfc__min_samples_leaf':[1, 5]
             }
    cv = GridSearchCV(pipe, param_grid=parameters, scoring='accuracy', verbose=1, n_jobs=1)                      
    return pipe


def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: List) -> None:
    '''
    Function for evaluating model by printing a classification report
    Args:   Model, features, labels to evaluate, and a list of categories
    Returns: Classification report
    '''
    y_pred = model.predict(X_test)                       
    
    print(classification_report(Y_test, y_pred, target_names = category_names))
    for  idx, cat in enumerate(Y_test.columns.values):
        print("{} -- {}".format(cat, accuracy_score(Y_test.values[:,idx], y_pred[:, idx])))
    print("accuracy = {}".format(accuracy_score(Y_test, y_pred)))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    '''
    Save the model as a pickle file
    ARGS: Model = the best Grid Search model
          filepath = the desired name of the pickle file
    OUTPUT: None. Saves model as a pickle file
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()