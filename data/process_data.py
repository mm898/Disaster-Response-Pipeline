import sys
import numpy as np
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    '''load the data

    Parameters:
    messages_filepath: the filepath for the messages

    Returns:
    df: a dataframe that needs to  be cleaned
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id', how='left')
    
    return df


def clean_data(df):
    '''clean the dublicate and split the data with semicolumns

    Parameters:
    df: the dataframe

    Returns:
    df: a cleaned dataframe without any duplicates
    '''

    # split the catergories with ;
    categories = df.categories.str.split(';', expand=True)
    
    row = categories.loc[0,:].values
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    '''save a dataframe as SQL database

    Parameters:
    df: the dataframe
    database_filename: the database name
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index=False)


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