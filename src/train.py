# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
from joblib import dump, load
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, average_precision_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection  import GridSearchCV




def LoadData(dataBasePath, databaseTableName):
    '''Loads the data from the database. Returns a pandas dataframe.'''
    # load data from database
    engine = create_engine(dataBasePath)
    df = pd.read_sql(databaseTableName, engine)
    return df

def PreprocessData(df):
    '''Preprocesses a pandas dataframe. Outputs features X and targets Y.'''
    X = df['message']
    Y = df.drop(columns=['id', 'message','original', 'genre'])
    
    #Remove classes that are not binary
    columns_to_remove = []
    for name in Y.columns:
        if(Y[name].min() != 0 or
            Y[name].max() != 1):
            print(f'This is not a binary feature, will remove it: {name}')
            columns_to_remove.append(name)


    if len(columns_to_remove) > 0:
        print(f'dropping {len(columns_to_remove)} Columns')
        Y.drop(columns=columns_to_remove, inplace=True)
        
    return X, Y

def tokenize(text):
    '''This will take a text string and tokenize it. A list of clean tokens are returned.'''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(x).lower().strip() for x in tokens]
    return clean_tokens

def CreatePipeline():
    '''This creates and returns a full pipeline object.'''
    pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('hashing_vectorizer', HashingVectorizer())
            ])),
           
            ('clf', MultiOutputClassifier(SGDClassifier(penalty='l2',
                                                        loss='log',
                                                        alpha=1e-5, random_state=42,
                                                        max_iter=5, tol=None)
            ))])


    #Create parameters to use during grid search
    parameters = {    
        'clf__estimator__loss': ['hinge', 'log', 'perceptron'],
        'clf__estimator__alpha': [1e-3, 1e-4, 1e-5],
        'clf__estimator__penalty': ['l1', 'l2'],
    }

    #Create a grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2, scoring='f1_weighted')
    return cv

def EvaluateModel(model, X_test, Y_test):
    '''This takes a model as input, along with the X and Y test sets. It then evaluates the model and outputs some metrics.'''
    Y_pred = model.predict(X_test)
    
    total_hits = np.sum(np.sum(Y_pred == Y_test))
    total_misses = np.sum(np.sum(Y_pred != Y_test))
    total_accuracy = total_hits/(total_hits + total_misses)
       
    target_names = [name for name in Y_test.columns]

    precisions = []
    recalls = []
    f1scores = []
    for (name, col) in zip(target_names, range(len(target_names))):
        y_test = Y_test[name].values
        y_pred = Y_pred[:, col]
        
        if(np.max(y_test) <= 1):
            #Only one category
            precisions.append(precision_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            f1scores.append(f1_score(y_test, y_pred))
        print(f'Category: {name}')
        print(classification_report(y_test, y_pred))
        print('-'*42)
     
    
    print(f'Total Accuracy: {total_accuracy}')
    print(f'Average Precission: {np.average(precisions)}')
    print(f'Average Recall: {np.average(recalls)}')
    print(f'Average F1 Score: {np.average(f1scores)}')
    
def main(modelName, dataBasePath, databaseTableName):    
    '''This runs the full train pipeline and writes a pickled model.'''
    #Load data
    print(f'Loading data from db: {dataBasePath} table: {databaseTableName}')
    df = LoadData(dataBasePath, databaseTableName)
    
    #Create Features/Targets
    print('Preprocessing data')
    X, Y = PreprocessData(df)
    
    #Split into Train/test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    #Create model pipeline
    print('Creating pipeline')
    pipeline = CreatePipeline()
    
    #Fit to data
    print('Training model')
    pipeline.fit(X_train, Y_train)
    
    print('Evaluating model...')
    EvaluateModel(pipeline, X_test, Y_test)
    
    #Save model to file
    print(f'Saving model as: {modelName}')
    dump(pipeline, modelName) 
    
if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='Trains a model to classify disaster responses.')
    parser.add_argument('--modelName', help='Name of the model', default='model.pkl')
    parser.add_argument('--dataBasePath', help='Path to the database', default='sqlite:///Data/DisasterResponse.db')
    parser.add_argument('--databaseTableName', help='Name of the database table', default='messagesDataTable')

    args = parser.parse_args()
    print('Training Model...')

    main(args.modelName, args.dataBasePath, args.databaseTableName)
    
    print('Program terminated...')

    