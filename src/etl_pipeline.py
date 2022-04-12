# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# App settings
messagesPath = 'Data/messages.csv' # Path to the messages data file
categoriesPath = 'Data/categories.csv' # Path to the categories data file
dataBasePath = 'sqlite:///Data/DisasterResponse.db'
databaseTableName = 'messagesDataTable'


def LoadData(messagesPath, categoriesPath):
    # load messages dataset
    messages = pd.read_csv(messagesPath)
    
    # load categories dataset
    categories = pd.read_csv(categoriesPath)
    
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    
    return df
    
def PreprocessData(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:str(x).split('-')[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        try:
            categories[column] = categories[column].apply(lambda x:str(x).split('-')[-1])

            # convert column from string to numeric
            categories[column] = categories[column].astype('int32')

        except Exception as e:
            print(e)
        
    
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True, how = 'left')
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
    
def SaveToDatabase(df, dataBasePath, databaseTableName):
    engine = create_engine(dataBasePath)
    df.to_sql(databaseTableName, engine, index=False,if_exists='replace')
    
    
def main(messagesPath, categoriesPath, dataBasePath, databaseTableName):
    print(f'Loading data. Messages: {messagesPath} Categories: {categoriesPath}')
    df = LoadData(messagesPath, categoriesPath)
    
    print('Preprocessing data...')
    df = PreprocessData(df)
    
    print(f'Saving data to database. Database path: {dataBasePath} Table Name: {databaseTableName}')
    SaveToDatabase(df, dataBasePath, databaseTableName)
    
if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='ETL pipeline to preprocess disaster response messages.')
    parser.add_argument('--messagesPath', help='Path to the messages csv file', default='Data/messages.csv')
    parser.add_argument('--categoriesPath', help='Path to the categories csv file.', default='Data/categories.csv')
    parser.add_argument('--dataBasePath', help='Path to the database', default='sqlite:///Data/DisasterResponse.db')
    parser.add_argument('--databaseTableName', help='Name of the database table', default='messagesDataTable')

    args = parser.parse_args()
    print('Running ETL Pipeline...')

    main(args.messagesPath, args.categoriesPath, args.dataBasePath, args.databaseTableName)
    
    print('Program terminated...')  
    
    
    
    
    
    